# SecureRAG

## Architecture and request flow

The FastAPI backend authenticates users with a JWT, reads the user's active role from PostgreSQL, embeds each question with `all-MiniLM-L6-v2`, and retrieves only Qdrant chunks allowed for that role. Authorized context and the user's recent questions go to Groq; the response is stored in that user's conversation. The role hierarchy is `intern` < `employee` < `manager` < `board`.

The Bunker stores role-labelled files under generated IDs. PDFs, plain text, Markdown, and quick notices become OKF Markdown and searchable Qdrant chunks. Images and short videos are stored and access-controlled in the Bunker; they are not interpreted by the chatbot. Filesystem status and Qdrant payload status prevent processing or failed documents from being treated as searchable.

## Setup

Use Python 3.11 or newer (the dependencies support Python 3.13):

```powershell
py -3.13 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r Backend/requirements.txt
```

Create `Backend/.env` with these variables (use your own values):

```dotenv
DATABASE_URL=postgresql://USER:PASSWORD@HOST:5432/DATABASE
JWT_SECRET=replace-with-a-long-random-secret
GROQ_API_KEY=replace-with-your-groq-api-key
QDRANT_URL=https://YOUR-CLUSTER.qdrant.io
QDRANT_API_KEY=replace-with-your-qdrant-api-key
```

The SQLAlchemy URL shown uses the PostgreSQL `psycopg2` driver included in the requirements. PostgreSQL must be reachable and the configured user must be able to create the application's tables. Qdrant must be reachable; the backend creates the `company_documents` collection and access-level indexes if needed. It does not delete or recreate an existing collection. The embedding model is loaded on first retrieval or ingestion, not during module import.

## Run

From the repository root:

```powershell
python -m uvicorn main:app --app-dir Backend --reload
```

Open `http://127.0.0.1:8000/` for the company home page and workspace.

The database needs a pre-provisioned user row with a bcrypt `password_hash` and one of the four supported roles. Sign in at `POST /login` with a JSON body containing `email` and `password`, then send the returned bearer token to `/chat` or `/bunker/upload`.

Create an account interactively (the password is entered without echo and stored only as a bcrypt hash):

```powershell
python Backend/create_user.py
```

Use the same company email and password on the login page. There is no public sign-up form; account creation is an administrator-run command.

## RBAC and document lifecycle

A user can read documents at or below their role's tier. The upload policy follows the same hierarchy: interns may assign `intern`; employees may assign `intern` or `employee`; managers may assign through `manager`; board users may assign any tier. The server gets the role from the active database user row, not from the JWT role claim or request form.

`POST /bunker/upload` accepts multipart fields `file` and `access_level`. It supports PDF, UTF-8 TXT/Markdown, PNG/JPEG/GIF/WebP, and MP4/MOV/WebM. Documents and images are limited to 20 MiB; videos to 100 MiB. PDF/text and quick notices created in the Bunker are searchable; images/videos are stored but not sent to Scout. The library and file route apply the same access hierarchy. The generated document ID is used for filesystem names and stable Qdrant chunk IDs. Files, OKF Markdown, and status JSON are kept in `Backend/uploads/`. Filesystem and Qdrant writes cannot be one atomic transaction; detected failures are marked failed and partial vectors are removed where possible. A service outage during cleanup may leave non-searchable processing vectors for operator review.

Existing OKF files under `documents/okf/` can be ingested with:

```powershell
python Backend/ingest.py
```

A one-off local search requires an explicit operator role and uses the same filter:

```powershell
python Backend/search.py --role intern "How many days of leave do employees get?"
```

## Tests and checks

Run mocked unit tests without live service credentials:

```powershell
python -m unittest discover -s Backend/tests -v
```

Check installed dependency consistency and Python syntax:

```powershell
python -m pip check
python -m compileall Backend
```
