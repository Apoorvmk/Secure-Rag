# SecureRAG

SecureRAG is a role-aware company knowledge assistant. It uses an existing Groq LLM; it does not train a model. The FastAPI app serves the plain JavaScript frontend from the same origin, stores users and chat records in PostgreSQL, stores document chunks in Qdrant, and optionally uses Redis for retrieval-result caching.

## Security model and implemented flow

- Roles are `intern` < `employee` < `manager` < `board`. Each document has one access label. The active role is read from PostgreSQL for every authenticated request.
- Login uses bcrypt password verification and a two-hour HS256 JWT. Protected routes reject missing, invalid, expired, disabled-user, or unknown-role credentials.
- PDF, UTF-8 TXT, Markdown, and quick notices are converted to OKF-style Markdown, chunked, embedded with `all-MiniLM-L6-v2`, and indexed in a shared Qdrant collection. Processing and failed points are excluded; only points explicitly marked `searchable` are retrieved.
- Upload retries with identical bytes and the same access label reuse the original document record. Reusing identical bytes under a different label is rejected. Chunk IDs are stable per document and chunk index.
- Search runs vector retrieval and BM25 over the same Qdrant-side role/status filter, then combines ranks with Reciprocal Rank Fusion. Results are checked against the role policy again in application code. BM25 scans at most 5,000 authorized chunks per query; this is intended for a small internal library and should be replaced with an indexed lexical backend for larger corpora.
- Context-dependent queries can be rewritten using recent user questions, while the original question is always retrieved too. Weak evidence triggers one corrective rewrite/search attempt. A lexical overlap or vector-score heuristic gates answer generation; when evidence remains weak, the API returns an uncertainty response instead of asking the LLM to guess.
- Answers are grounded in authorized chunks and include source references (`[S1]`, etc.) with document metadata. Document text is treated as untrusted input. The API does not pass previous assistant answers into a later retrieval prompt.
- A cross-encoder reranker is supported and safely skipped if its model cannot load. Set `RERANKER_MODEL` to a Sentence Transformers cross-encoder model ID to enable it; the model is downloaded on first use if it is not cached locally.
- Optional Redis caching uses a role-specific key, query digest, shared index epoch, and 45-second TTL. Successful indexing advances the epoch. If Redis is unavailable, retrieval runs directly against Qdrant. Use a private, authenticated/TLS Redis endpoint in deployments; the local Compose Redis is intentionally internal and non-persistent.
- Structured audit events omit passwords, tokens, user question text, file names, and document contents. `/metrics` exposes Prometheus-compatible HTTP and retrieval/ingestion stage counters and timings. Metrics are in-process and are intended for a single API worker; use an external aggregation strategy before running multiple workers.
- Images and videos are stored and access-controlled in Bunker but are not OCR'd, transcribed, or searchable. Scanned PDFs with no extractable text are rejected; OCR is not implemented.

## Local setup with Docker Compose

Install Docker Desktop, then from the repository root:

```powershell
Copy-Item .env.example .env
```

Edit `.env`: set a strong random `JWT_SECRET`, a Groq API key, a local Postgres password, and a strong `GRAFANA_ADMIN_PASSWORD`. `QDRANT_API_KEY` has a local development value in the example; change it for a shared environment. Obtain `GROQ_API_KEY` from your Groq account. Never commit `.env`.

Start the backend and its data services:

```powershell
docker compose up --build
```

Open `http://localhost:8000/`. Create the first company account in another PowerShell window:

```powershell
docker compose exec api python Backend/create_user.py
```

The password is prompted without echo and stored as a bcrypt hash. There is no public signup; this administrator-only command explicitly activates the account. Login rejects inactive or unset account status. Existing databases must have `users.is_active` set to `NOT NULL DEFAULT false`, and intended accounts must be activated explicitly; SQLAlchemy `create_all` does not migrate existing tables. The API creates missing SQLAlchemy tables on first database use. Qdrant collections and indexes are created lazily when first used. First ingestion/query downloads the sentence embedding model if it is not cached.

To start the prebuilt monitoring profile as well:

```powershell
docker compose --profile observability up --build
```

Prometheus is at `http://localhost:9090`; Grafana is at `http://localhost:3000` with the admin credentials from `.env`. The provisioned dashboard is named **SecureRAG Overview**. This configuration is a local development setup, not a production hardening guide.

## Run without a containerized API

Create local PostgreSQL, Qdrant, and optionally Redis services. Copy `Backend/.env.example` to `Backend/.env`, set service URLs and a strong JWT secret, then:

```powershell
py -3.13 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r Backend/requirements.txt
python -m uvicorn main:app --app-dir Backend --reload
```

Visit `http://127.0.0.1:8000/`. The default API URL is same-origin and needs no frontend build. If the UI should call a separately hosted API, set `API_BASE_URL` to its origin and set `CORS_ORIGINS` to the exact comma-separated browser-origin allowlist. Do not use `*` with credentials.

## Evaluation

Run the safe synthetic BM25 evaluation locally, without PostgreSQL, Qdrant, Redis, Groq, or company documents:

```powershell
python Backend/evaluate.py --dataset Backend/eval/sample_dataset.json --k 5
```

The JSON dataset format has a `corpus` array (`id`, `access_level`, `text`) and `queries` array (`question`, `role`, `relevant_ids`, optional `reference_answer`). The output includes measured Recall@K, MRR, mean local query latency, and a lexical reference-answer evidence-coverage proxy. The proxy is **not** a semantic faithfulness score or an LLM judge. A query with no authorized expected documents is excluded from Recall@K/MRR and shown with `null` Recall.

For a supplied answer to assess, a query may also include `candidate_answer`; the evaluator reports token overlap between that answer and retrieved evidence as `candidate_answer_evidence_support_proxy`. This is a lightweight grounding proxy, not a claim-level or semantic faithfulness metric. Live evaluation reports measured latency by retrieval stage when that stage ran.

To evaluate real Qdrant hybrid retrieval, supply a dataset whose relevant IDs match the indexed Qdrant `document_id` values and run:

```powershell
python Backend/evaluate.py --live --dataset path\to\your-eval.json --k 5
```

Live mode requires configured Qdrant and the embedding model, but does not call Groq. Use synthetic or approved non-confidential evaluation data. The included sample corpus is not production performance data.

## Tests and checks

```powershell
python -m unittest discover -s Backend/tests -v
python -m pip check
node --check Frontend/app.js
```

Tests use local fakes/mocks for PostgreSQL, Qdrant, Redis, and Groq behavior; they do not verify real service connectivity, model downloads, Groq answer quality, or Docker image builds. GitHub Actions runs the unit suite and basic Python/JavaScript/dependency checks.

## Operational limits

- No Alembic migrations are configured. SQLAlchemy creates missing tables but does not migrate changed table schemas.
- Legacy Qdrant points without `ingestion_status: searchable` are intentionally excluded. Re-ingest existing OKF documents with `python Backend/ingest.py` after configuring the new version.
- Files and per-document status metadata are stored under `Backend/uploads/`; Docker Compose persists this folder in a named volume. Back it up alongside PostgreSQL and Qdrant.
- Filesystem and Qdrant do not share an atomic transaction. Failed writes are marked non-searchable and partial points are deleted when possible; a Qdrant outage during cleanup or a process crash that leaves a `processing` record can require operator review.
- OCR, image understanding, video transcription, generated-answer faithfulness judging, and production-scale BM25 indexing are not implemented.
- Compose and Grafana provisioning are configuration artifacts. They have not been validated against a live Docker daemon or external services in this environment.
