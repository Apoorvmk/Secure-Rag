import asyncio
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from access_policy import allowed_access_levels, can_upload_to_level
from document_pipeline import (DocumentValidationError, filter_authorized_results, ingest_markdown,
                               make_okf_markdown, parse_okf, pdf_to_text, retrieve_authorized, chunk_text)
from rag_pipeline import (bm25_rank, evidence_is_sufficient, reciprocal_rank_fusion,
                          retrieve_hybrid, rewrite_query)


class Vector(list):
    def tolist(self):
        return list(self)


class Model:
    def encode(self, value):
        if isinstance(value, str):
            return Vector([0.1, 0.2])
        return [Vector([float(i), 0.2]) for i, _ in enumerate(value)]


class FakeQdrant:
    def __init__(self):
        self.points = {}
        self.filters = []
        self.fail_upsert = False

    def upsert(self, *, points, **kwargs):
        if self.fail_upsert:
            raise RuntimeError("external error with secret-like value")
        self.points.update({str(point.id): point for point in points})

    def scroll(self, *, scroll_filter, **kwargs):
        doc_id = scroll_filter.must[0].match.value
        points = [point for point in self.points.values()
                  if point.payload.get("document_id") == doc_id]
        return points, None

    def delete(self, *, points_selector, **kwargs):
        if isinstance(points_selector, list):
            for point_id in points_selector:
                self.points.pop(str(point_id), None)
        else:
            doc_id = points_selector.must[0].match.value
            self.points = {key: value for key, value in self.points.items()
                           if value.payload.get("document_id") != doc_id}

    def set_payload(self, *, payload, points, **kwargs):
        doc_id = points.must[0].match.value
        for point in self.points.values():
            if point.payload.get("document_id") == doc_id:
                point.payload.update(payload)

    def query_points(self, **kwargs):
        self.filters.append(kwargs["query_filter"])
        return SimpleNamespace(points=[])


def okf(document_id="doc-1", text="A" * 700, level="intern"):
    return make_okf_markdown(text, title="Policy", access_level=level,
                             source="policy.pdf", document_id=document_id, uploaded_by=3)


class AccessPolicyTests(unittest.TestCase):
    def setUp(self):
        # Keep ingestion tests independent from any REDIS_URL in a developer's .env.
        import retrieval_cache
        redis_patch = patch.object(retrieval_cache, "get_redis", return_value=None)
        redis_patch.start()
        self.addCleanup(redis_patch.stop)

    def test_visibility_and_upload_hierarchy(self):
        self.assertEqual(allowed_access_levels("intern"), ["intern"])
        self.assertEqual(allowed_access_levels("employee"), ["intern", "employee"])
        self.assertEqual(allowed_access_levels("manager"), ["intern", "employee", "manager"])
        self.assertEqual(allowed_access_levels("board"), ["intern", "employee", "manager", "board"])
        self.assertFalse(can_upload_to_level("intern", "board"))
        self.assertTrue(can_upload_to_level("employee", "employee"))
        self.assertFalse(can_upload_to_level("unknown", "intern"))

    def test_all_role_searches_apply_shared_filter(self):
        qdrant, model = FakeQdrant(), Model()
        for role in ("intern", "employee", "manager", "board"):
            retrieve_authorized("question", role=role, qdrant=qdrant, model=model, limit=3)
        levels = [flt.must[0].match.any for flt in qdrant.filters]
        self.assertEqual(levels, [
            ["intern"], ["intern", "employee"],
            ["intern", "employee", "manager"],
            ["intern", "employee", "manager", "board"]])
        self.assertEqual(qdrant.filters[0].must[1].match.value, "searchable")

    def test_qdrant_filter_enforces_hierarchy_and_ingestion_status(self):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams
        qdrant = QdrantClient(":memory:")
        qdrant.create_collection("company_documents", vectors_config=VectorParams(size=2, distance=Distance.COSINE))
        points = []
        for index, level in enumerate(("intern", "employee", "manager", "board")):
            points.append(PointStruct(id=index + 1, vector=[0.1, 0.2], payload={
                "access_level": level, "text": level, "ingestion_status": "searchable"}))
        points.append(PointStruct(id=5, vector=[0.1, 0.2], payload={
            "access_level": "board", "text": "processing secret", "ingestion_status": "processing"}))
        points.append(PointStruct(id=6, vector=[0.1, 0.2], payload={
            "access_level": "employee", "text": "legacy employee"}))
        qdrant.upsert("company_documents", points=points)
        expected = {
            "intern": {"intern"},
            "employee": {"intern", "employee"},
            "manager": {"intern", "employee", "manager"},
            "board": {"intern", "employee", "manager", "board"},
        }
        for role, allowed in expected.items():
            results = retrieve_authorized("question", role=role, qdrant=qdrant, model=Model(), limit=10)
            self.assertEqual({point.payload["text"] for point in results}, allowed)
        qdrant.close()

    def test_untrusted_or_missing_payload_metadata_is_dropped(self):
        points = [SimpleNamespace(payload={"access_level": "intern", "text": "safe", "ingestion_status": "searchable"}),
                  SimpleNamespace(payload={"access_level": "board", "text": "secret", "ingestion_status": "searchable"}),
                  SimpleNamespace(payload={"access_level": ["intern"], "text": "malformed ACL", "ingestion_status": "searchable"}),
                  SimpleNamespace(payload={"text": "missing ACL", "ingestion_status": "searchable"}),
                  SimpleNamespace(payload=None)]
        visible = filter_authorized_results(points, "intern")
        self.assertEqual([point.payload["text"] for point in visible], ["safe"])

    def test_reingest_uses_stable_ids_and_single_vector_set(self):
        qdrant, model = FakeQdrant(), Model()
        content = okf()
        first_count = ingest_markdown(content, qdrant=qdrant, model=model)
        first_ids = set(qdrant.points)
        second_count = ingest_markdown(content, qdrant=qdrant, model=model)
        self.assertEqual(first_count, second_count)
        self.assertEqual(first_ids, set(qdrant.points))
        self.assertEqual(len(qdrant.points), 2)
        shorter = okf(text="short policy")
        ingest_markdown(shorter, qdrant=qdrant, model=model)
        self.assertEqual(len(qdrant.points), 1)
        for point in qdrant.points.values():
            self.assertEqual(point.payload["access_level"], "intern")
            self.assertEqual(point.payload["document_id"], "doc-1")
            self.assertIn("source", point.payload)
            self.assertIn("title", point.payload)
            self.assertEqual(point.payload["uploaded_by"], 3)
            self.assertEqual(point.payload["ingestion_status"], "searchable")

    def test_partial_upsert_failure_cleans_document_vectors(self):
        qdrant = FakeQdrant()
        qdrant.fail_upsert = True
        with self.assertRaises(RuntimeError):
            ingest_markdown(okf(), qdrant=qdrant, model=Model())
        self.assertEqual(qdrant.points, {})


class HybridRetrievalTests(unittest.TestCase):
    def test_bm25_and_rrf_rank_relevant_evidence(self):
        corpus = [
            {"id": "a", "text": "Employees receive twenty paid leave days each year."},
            {"id": "b", "text": "Managers approve department travel expenses."},
        ]
        ranked = bm25_rank("How many paid leave days do employees get", corpus, limit=2)
        self.assertEqual(ranked[0]["id"], "a")
        fused = reciprocal_rank_fusion(
            [SimpleNamespace(id="b", payload={"text": "b"}, score=0.8),
             SimpleNamespace(id="a", payload={"text": "a"}, score=0.5)],
            [SimpleNamespace(id="a", payload={"text": "a"}, score=None)], limit=2)
        self.assertEqual(fused[0].id, "a")

    def test_hybrid_retrieval_with_local_qdrant_collection(self):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams
        qdrant = QdrantClient(":memory:")
        qdrant.create_collection("company_documents", vectors_config=VectorParams(size=2, distance=Distance.COSINE))
        qdrant.upsert("company_documents", points=[
            PointStruct(id=1, vector=[0.9, 0.1], payload={"text": "Interns request paid leave through their supervisor.",
                "access_level": "intern", "ingestion_status": "searchable", "document_id": "intern-doc"}),
            PointStruct(id=2, vector=[0.8, 0.2], payload={"text": "Employees receive 20 paid leave days.",
                "access_level": "employee", "ingestion_status": "searchable", "document_id": "employee-doc"}),
            PointStruct(id=3, vector=[1.0, 0.0], payload={"text": "Board restricted leave policy.",
                "access_level": "board", "ingestion_status": "searchable", "document_id": "board-doc"}),
            PointStruct(id=4, vector=[1.0, 0.0], payload={"text": "Employee processing leave policy.",
                "access_level": "employee", "ingestion_status": "processing", "document_id": "pending-doc"}),
        ])
        results = retrieve_hybrid("paid leave", role="intern", qdrant=qdrant, model=Model(), limit=4)
        self.assertEqual([item.payload["document_id"] for item in results], ["intern-doc"])
        qdrant.close()

    def test_hybrid_vector_and_keyword_queries_share_role_and_status_filter(self):
        from access_policy import authorized_qdrant_filter

        class SearchQdrant:
            def __init__(self):
                self.query_filter = None
                self.scroll_filter = None
                self.points = [
                    SimpleNamespace(id="int", score=0.4, payload={
                        "text": "Intern orientation covers mentorship and onboarding.",
                        "access_level": "intern", "ingestion_status": "searchable"}),
                    SimpleNamespace(id="emp", score=0.5, payload={
                        "text": "Employee leave policy grants paid leave days.",
                        "access_level": "employee", "ingestion_status": "searchable"}),
                    SimpleNamespace(id="board", score=0.9, payload={
                        "text": "Board strategy is confidential.", "access_level": "board",
                        "ingestion_status": "searchable"}),
                    SimpleNamespace(id="pending", score=1.0, payload={
                        "text": "Employee pending leave secret.", "access_level": "employee",
                        "ingestion_status": "processing"}),
                ]

            def query_points(self, **kwargs):
                self.query_filter = kwargs["query_filter"]
                return SimpleNamespace(points=self.points)

            def scroll(self, **kwargs):
                self.scroll_filter = kwargs["scroll_filter"]
                levels = set(kwargs["scroll_filter"].must[0].match.any)
                points = [point for point in self.points
                          if point.payload.get("access_level") in levels
                          and point.payload.get("ingestion_status") == "searchable"]
                return points, None

        qdrant = SearchQdrant()
        results = retrieve_hybrid("employee paid leave", role="employee", qdrant=qdrant,
                                  model=Model(), limit=3)
        self.assertEqual(qdrant.query_filter, authorized_qdrant_filter("employee"))
        self.assertEqual(qdrant.scroll_filter, qdrant.query_filter)
        self.assertEqual(results[0].payload["access_level"], "employee")
        self.assertNotIn("board", [point.id for point in results])
        self.assertNotIn("pending", [point.id for point in results])

    def test_reranker_and_rewrite_failures_fall_back_safely(self):
        class BrokenReranker:
            def predict(self, _pairs):
                raise RuntimeError("unavailable")
        class SearchQdrant:
            def query_points(self, **kwargs):
                return SimpleNamespace(points=[])
            def scroll(self, **kwargs):
                return [], None
        self.assertEqual(retrieve_hybrid("question", role="intern", qdrant=SearchQdrant(),
            model=Model(), limit=2, reranker=BrokenReranker()), [])
        self.assertIsNone(rewrite_query("And what about it?", ["leave policy"], None))

        points = [SimpleNamespace(id="first", score=0.8, payload={"text": "alpha policy",
                      "access_level": "intern", "ingestion_status": "searchable"}),
                  SimpleNamespace(id="second", score=0.7, payload={"text": "beta policy",
                      "access_level": "intern", "ingestion_status": "searchable"})]
        class RankedQdrant:
            def query_points(self, **_kwargs): return SimpleNamespace(points=points)
            def scroll(self, **_kwargs): return points, None
        class WorkingReranker:
            def predict(self, pairs): return [0.0 if "alpha" in text else 1.0 for _, text in pairs]
        ranked = retrieve_hybrid("policy", role="intern", qdrant=RankedQdrant(),
            model=Model(), limit=2, reranker=WorkingReranker())
        self.assertEqual(ranked[0].id, "second")

    def test_contextual_query_rewrite_and_evidence_refusal_decision(self):
        completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content="employee leave policy days"))])
        llm = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **_kwargs: completion)))
        rewritten = rewrite_query("And how many does it allow?", ["employee leave policy"], llm)
        self.assertEqual(rewritten, "employee leave policy days")
        good = [SimpleNamespace(payload={"text": "Employees receive twenty paid leave days."}, semantic_score=None)]
        self.assertTrue(evidence_is_sufficient("paid leave days", good))
        self.assertFalse(evidence_is_sufficient("secret merger plan", []))


class CacheAndMetricsTests(unittest.TestCase):
    def test_redis_cache_scopes_role_and_epoch_and_fails_open(self):
        import retrieval_cache

        class FakeRedis:
            def __init__(self): self.values = {}
            def get(self, key): return self.values.get(key)
            def set(self, key, value, nx=False): self.values.setdefault(key, value)
            def setex(self, key, _ttl, value): self.values[key] = value
            def incr(self, key):
                self.values[key] = str(int(self.values.get(key, "0")) + 1)
                return int(self.values[key])
        fake = FakeRedis()
        point = SimpleNamespace(id="chunk-1", payload={"text": "allowed",
            "access_level": "intern", "ingestion_status": "searchable"}, score=0.5,
            semantic_score=0.4)
        with patch.object(retrieval_cache, "get_redis", return_value=fake):
            retrieval_cache.set_cached("intern", "policy", [point])
            retrieval_cache.set_cached("board", "policy", [point])
            self.assertEqual(retrieval_cache.get_cached("intern", "policy")[0].id, "chunk-1")
            self.assertEqual(retrieval_cache.get_cached("board", "policy")[0].id, "chunk-1")
            self.assertTrue(retrieval_cache.invalidate_retrieval_cache())
            self.assertIsNone(retrieval_cache.get_cached("intern", "policy"))
        with patch.object(retrieval_cache, "get_redis", return_value=None):
            self.assertIsNone(retrieval_cache.get_cached("intern", "policy"))
        class UnavailableRedis:
            def get(self, *_args): raise ConnectionError("offline")
        with patch.object(retrieval_cache, "get_redis", return_value=UnavailableRedis()):
            self.assertIsNone(retrieval_cache.get_cached("intern", "policy"))
            retrieval_cache.set_cached("intern", "policy", [point])
            self.assertFalse(retrieval_cache.invalidate_retrieval_cache())

    def test_audit_event_and_prometheus_pipeline_metrics(self):
        from observability import audit_event, prometheus_text, record_stage
        with self.assertLogs("securerag.audit", level="INFO") as captured:
            audit_event("login", outcome="success", user_id=4, role="employee")
        self.assertIn('"event":"login"', captured.output[0])
        self.assertNotIn("password", captured.output[0])
        record_stage("retrieval", 0.025)
        self.assertIn("securerag_pipeline_stage_total", prometheus_text())


class EvaluationTests(unittest.TestCase):
    def test_safe_sample_evaluation_reports_real_local_metrics_and_access_scope(self):
        import evaluate
        dataset = json.loads((Path(__file__).resolve().parents[1] / "eval" /
                              "sample_dataset.json").read_text(encoding="utf-8"))
        result = evaluate.run_local(dataset, k=3)
        self.assertEqual(result["mode"], "local_bm25_synthetic_or_user_supplied_corpus")
        self.assertEqual(result["queries"], 3)
        self.assertEqual(result["queries_with_authorized_relevant_documents"], 2)
        self.assertEqual(result["recall_at_k"], 1.0)
        intern_case = result["details"][2]
        self.assertEqual(intern_case["ranked_ids"], [])
        self.assertIsNone(intern_case["recall_at_k"])


class ConfigurationTests(unittest.TestCase):
    def test_compose_monitoring_and_ci_configuration_parse(self):
        import yaml
        root = Path(__file__).resolve().parents[2]
        compose = yaml.safe_load((root / "docker-compose.yml").read_text(encoding="utf-8"))
        scrape = yaml.safe_load((root / "ops" / "prometheus.yml").read_text(encoding="utf-8"))
        datasource = yaml.safe_load((root / "ops" / "grafana" / "provisioning" /
                                     "datasources" / "prometheus.yml").read_text(encoding="utf-8"))
        json.loads((root / "ops" / "grafana" / "dashboards" / "securerag.json").read_text(encoding="utf-8"))
        self.assertIn("api", compose["services"])
        self.assertIn("redis", compose["services"])
        self.assertEqual(scrape["scrape_configs"][0]["metrics_path"], "/metrics")
        self.assertEqual(datasource["datasources"][0]["uid"], "Prometheus")


class DocumentFormatTests(unittest.TestCase):
    def test_pdf_extraction_okf_metadata_and_chunking(self):
        import pymupdf
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "policy.pdf"
            document = pymupdf.open()
            page = document.new_page()
            page.insert_text((72, 72), "Employees receive paid leave.")
            document.save(path)
            document.close()
            extracted = pdf_to_text(path)
        self.assertIn("Employees receive paid leave", extracted)
        markdown = make_okf_markdown(extracted, title="Leave Policy", access_level="employee",
            source="policy.pdf", document_id="doc-7", uploaded_by=5)
        metadata, body = parse_okf(markdown)
        self.assertEqual(metadata["document_id"], "doc-7")
        self.assertEqual(metadata["access_level"], "employee")
        self.assertIn("Employees receive paid leave", body)
        self.assertEqual(chunk_text("abcdefgh", size=3), ["abc", "def", "gh"])


class UploadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import main
            cls.main = main
        except RuntimeError as exc:
            raise unittest.SkipTest(f"API dependencies unavailable: {exc}")

    def call_upload(self, data, filename="policy.pdf", role="employee", level="employee"):
        from fastapi import UploadFile
        upload = UploadFile(filename=filename, file=__import__("io").BytesIO(data))
        return asyncio.run(self.main.upload_document(upload, level, {"user_id": 3, "role": role}))

    def test_upload_ingests_and_marks_searchable_with_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            self.main.BUNKER_DIR = Path(folder)
            with patch.object(self.main, "pdf_to_text", return_value="policy text"), \
                 patch.object(self.main, "make_okf_markdown", return_value="okf"), \
                 patch.object(self.main, "ingest_markdown", return_value=1) as ingest, \
                 patch.object(self.main, "get_qdrant", return_value=object()), \
                 patch.object(self.main, "get_embedding_model", return_value=object()):
                result = self.call_upload(b"%PDF-valid")
            self.assertEqual(result["status"], "searchable")
            record = json.loads((Path(folder) / f"{result['document_id']}.json").read_text())
            self.assertEqual(record["status"], "searchable")
            self.assertEqual(record["access_level"], "employee")
            ingest.assert_called_once()

    def test_text_upload_and_quick_notice_are_searchable(self):
        with tempfile.TemporaryDirectory() as folder:
            self.main.BUNKER_DIR = Path(folder)
            with patch.object(self.main, "ingest_markdown", return_value=1) as ingest, \
                 patch.object(self.main, "get_qdrant", return_value=object()), \
                 patch.object(self.main, "get_embedding_model", return_value=object()):
                uploaded = self.call_upload(b"A short searchable notice.", filename="notice.txt")
                from main import NoteRequest
                note = self.main.create_bunker_note(
                    NoteRequest(title="Office hours", text="The office closes early Friday.", access_level="intern"),
                    {"user_id": 3, "role": "employee"})
            self.assertTrue(uploaded["searchable"])
            self.assertTrue(note["searchable"])
            self.assertEqual(ingest.call_count, 2)
            self.assertTrue((Path(folder) / f"{uploaded['document_id']}.md").exists())

    def test_images_and_video_are_stored_but_not_claimed_as_searchable(self):
        with tempfile.TemporaryDirectory() as folder:
            self.main.BUNKER_DIR = Path(folder)
            image = self.call_upload(b"\x89PNG\r\n\x1a\nimage-data", filename="team.png")
            video = self.call_upload(b"\x00\x00\x00\x18ftypisomvideo", filename="update.mp4")
            self.assertEqual(image["status"], "stored")
            self.assertEqual(video["status"], "stored")
            self.assertFalse(image["searchable"])
            self.assertFalse(video["searchable"])
            self.assertTrue((Path(folder) / f"{image['document_id']}.png").exists())
            self.assertTrue((Path(folder) / f"{video['document_id']}.mp4").exists())

    def test_bunker_listing_and_file_route_obey_access_label(self):
        with tempfile.TemporaryDirectory() as folder:
            self.main.BUNKER_DIR = Path(folder)
            intern_id, board_id = "11111111-1111-4111-8111-111111111111", "22222222-2222-4222-8222-222222222222"
            for document_id, level in ((intern_id, "intern"), (board_id, "board")):
                (Path(folder) / f"{document_id}.json").write_text(json.dumps({
                    "document_id": document_id, "original_filename": "policy.txt", "title": "Policy",
                    "category": "document", "stored_extension": ".txt", "access_level": level,
                    "status": "stored", "searchable": False}), encoding="utf-8")
                (Path(folder) / f"{document_id}.txt").write_text("policy", encoding="utf-8")
            intern_items = self.main.list_bunker_documents({"user_id": 1, "role": "intern"})["documents"]
            self.assertEqual([item["access_level"] for item in intern_items], ["intern"])
            with self.assertRaises(Exception) as caught:
                self.main.get_bunker_file(board_id, {"user_id": 1, "role": "intern"})
            self.assertEqual(getattr(caught.exception, "status_code", None), 403)

    def test_backend_serves_company_home_and_frontend_assets(self):
        from fastapi.testclient import TestClient
        client = TestClient(self.main.app)
        home = client.get("/")
        script = client.get("/assets/app.js")
        metrics = client.get("/metrics")
        health = client.get("/health")
        runtime_config = client.get("/runtime-config.js")
        self.assertEqual(home.status_code, 200)
        self.assertIn("COMPANY WORKSPACE", home.text)
        self.assertEqual(script.status_code, 200)
        self.assertEqual(metrics.status_code, 200)
        self.assertIn("securerag_http_requests_total", metrics.text)
        self.assertEqual(runtime_config.status_code, 200)
        self.assertIn("window.SECURERAG_API_BASE_URL", runtime_config.text)
        with patch.dict(self.main.os.environ, {"API_BASE_URL": "https://api.example.test"}):
            configured = client.get("/runtime-config.js")
        self.assertIn('"https://api.example.test"', configured.text)
        self.assertEqual(health.json(), {"status": "ok"})

    def test_login_accepts_credentials_in_json_body(self):
        from fastapi.testclient import TestClient
        user = SimpleNamespace(id=3, is_active=True, role="employee", name="Ari", password_hash="hash")
        class Query:
            def filter(self, *_args): return self
            def first(self): return user
        class SessionStub:
            def __init__(self, _engine): pass
            def __enter__(self): return self
            def __exit__(self, *_args): pass
            def query(self, *_args): return Query()
        with patch.object(self.main, "get_engine", return_value=object()), \
             patch.object(self.main, "Session", SessionStub), \
             patch.object(self.main.bcrypt, "checkpw", return_value=True), \
             patch.object(self.main.jwt, "encode", return_value="test-token"):
            response = TestClient(self.main.app).post("/login", json={
                "email": "ari@northstar.example", "password": "secret-password"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["role"], "employee")
        self.assertEqual(response.json()["access_token"], "test-token")
        self.assertNotIn("password", str(response.url))

    def test_invalid_credentials_and_invalid_token_are_rejected(self):
        from fastapi.testclient import TestClient
        class Query:
            def filter(self, *_args): return self
            def first(self): return None
        class SessionStub:
            def __init__(self, _engine): pass
            def __enter__(self): return self
            def __exit__(self, *_args): pass
            def query(self, *_args): return Query()
        with patch.object(self.main, "get_engine", return_value=object()), \
             patch.object(self.main, "Session", SessionStub):
            client = TestClient(self.main.app)
            login = client.post("/login", json={"email": "missing@example.test", "password": "password"})
        self.assertEqual(login.status_code, 401)
        with patch.object(self.main, "_required_env", return_value="test-secret"):
            protected = TestClient(self.main.app).get("/me", headers={"Authorization": "Bearer not-a-jwt"})
        self.assertEqual(protected.status_code, 401)
        expired = self.main.jwt.encode({"user_id": 3,
            "exp": datetime.now(timezone.utc) - timedelta(minutes=1)}, "test-secret", algorithm="HS256")
        with patch.object(self.main, "_required_env", return_value="test-secret"):
            protected = TestClient(self.main.app).get("/me", headers={"Authorization": f"Bearer {expired}"})
        self.assertEqual(protected.status_code, 401)

    def test_chat_refuses_weak_evidence_and_does_not_replay_old_prompts(self):
        class Query:
            def __init__(self, model): self.model = model
            def filter(self, *_args): return self
            def order_by(self, *_args): return self
            def first(self): return None
            def all(self):
                if self.model is self.main.Message:
                    return [SimpleNamespace(content="board confidential acquisition plan")]
                return []
        class SessionStub:
            def __init__(self, _engine): self.rows = []
            def __enter__(self): return self
            def __exit__(self, *_args): pass
            def query(self, model): return Query(model)
            def add(self, row): self.rows.append(row)
            def flush(self):
                for row in self.rows:
                    if isinstance(row, self.main.Conversation): row.id = 88
            def commit(self): pass
        Query.main = self.main
        SessionStub.main = self.main
        with patch.object(self.main, "get_engine", return_value=object()), \
             patch.object(self.main, "Session", SessionStub), \
             patch.object(self.main, "cached_hybrid_retrieve", return_value=[]), \
             patch.object(self.main, "get_groq_client", return_value=None):
            refused = self.main.chat(self.main.ChatRequest(message="secret merger strategy"),
                {"user_id": 3, "role": "intern", "name": "Ari"})
        self.assertIn("can’t answer confidently", refused["response"])
        self.assertEqual(refused["sources"], [])

        safe_point = SimpleNamespace(id="intern-1", score=0.8, semantic_score=0.8,
            payload={"text": "Intern onboarding policy covers mentorship.",
                     "access_level": "intern", "ingestion_status": "searchable",
                     "title": "Intern guide", "source": "intern.md", "document_id": "doc-intern"})
        sent = {}
        answer = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Intern onboarding includes mentorship. [S1]"))])
        llm = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: (sent.update(kwargs) or answer))))
        with patch.object(self.main, "get_engine", return_value=object()), \
             patch.object(self.main, "Session", SessionStub), \
             patch.object(self.main, "cached_hybrid_retrieve", return_value=[safe_point]), \
             patch.object(self.main, "get_groq_client", return_value=llm):
            grounded = self.main.chat(self.main.ChatRequest(message="What does onboarding mentorship cover?"),
                {"user_id": 3, "role": "intern", "name": "Ari"})
        prompt_text = " ".join(message["content"] for message in sent["messages"])
        self.assertNotIn("board confidential", prompt_text)
        self.assertEqual(grounded["sources"][0]["document_id"], "doc-intern")

        with patch.object(self.main, "get_engine", return_value=object()), \
             patch.object(self.main, "Session", SessionStub), \
             patch.object(self.main, "measured_rewrite", side_effect=[None, "intern onboarding mentorship"]), \
             patch.object(self.main, "cached_hybrid_retrieve", side_effect=[[], [safe_point]]), \
             patch.object(self.main, "get_groq_client", return_value=llm):
            corrected = self.main.chat(self.main.ChatRequest(message="secret merger strategy"),
                {"user_id": 3, "role": "intern", "name": "Ari"})
        self.assertIn("Intern onboarding", corrected["response"])
        self.assertEqual(corrected["sources"][0]["document_id"], "doc-intern")
        self.assertIn("intern onboarding mentorship", " ".join(
            item["content"] for item in sent["messages"]))

    def test_identical_upload_retry_is_idempotent(self):
        with tempfile.TemporaryDirectory() as folder:
            self.main.BUNKER_DIR = Path(folder)
            with patch.object(self.main, "pdf_to_text", return_value="policy text"), \
                 patch.object(self.main, "make_okf_markdown", return_value="okf"), \
                 patch.object(self.main, "ingest_markdown", return_value=1) as ingest, \
                 patch.object(self.main, "get_qdrant", return_value=object()), \
                 patch.object(self.main, "get_embedding_model", return_value=object()):
                first = self.call_upload(b"%PDF-same-file")
                second = self.call_upload(b"%PDF-same-file")
            self.assertEqual(first["document_id"], second["document_id"])
            self.assertTrue(second["duplicate"])
            ingest.assert_called_once()
            with self.assertRaises(Exception) as caught:
                self.call_upload(b"%PDF-same-file", role="intern", level="intern")
            self.assertEqual(getattr(caught.exception, "status_code", None), 409)

    def test_identical_upload_does_not_duplicate_an_inflight_ingestion(self):
        with tempfile.TemporaryDirectory() as folder:
            self.main.BUNKER_DIR = Path(folder)
            content = b"%PDF-inflight"
            document_id = "33333333-3333-4333-8333-333333333333"
            record = {"document_id": document_id, "original_filename": "same.pdf",
                "access_level": "employee", "status": "processing",
                "sha256": __import__("hashlib").sha256(content).hexdigest()}
            (Path(folder) / f"{document_id}.json").write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaises(Exception) as caught:
                self.call_upload(content)
            self.assertEqual(getattr(caught.exception, "status_code", None), 409)

    def test_unauthenticated_upload_is_rejected(self):
        from fastapi.testclient import TestClient
        response = TestClient(self.main.app).post("/bunker/upload",
            data={"access_level": "intern"}, files={"file": ("x.pdf", b"%PDF-valid", "application/pdf")})
        self.assertIn(response.status_code, (401, 403))

    def test_database_role_overrides_stale_jwt_role(self):
        class Query:
            def filter(self, *_args):
                return self
            def first(self):
                return SimpleNamespace(id=3, is_active=True, role="intern", name="Ari")
        class SessionStub:
            def __init__(self, _engine): pass
            def __enter__(self): return self
            def __exit__(self, *_args): pass
            def query(self, *_args): return Query()
        credentials = SimpleNamespace(credentials="signed-token")
        with patch.object(self.main.jwt, "decode", return_value={"user_id": 3, "role": "board"}), \
             patch.object(self.main, "get_engine", return_value=object()), \
             patch.object(self.main, "Session", SessionStub):
            current = self.main.get_current_user(credentials)
        self.assertEqual(current, {"user_id": 3, "role": "intern", "name": "Ari"})

    def test_intern_cannot_request_board_access(self):
        with tempfile.TemporaryDirectory() as folder:
            self.main.BUNKER_DIR = Path(folder)
            with self.assertRaises(Exception) as caught:
                self.call_upload(b"%PDF-valid", role="intern", level="board")
            self.assertEqual(getattr(caught.exception, "status_code", None), 403)
            self.assertFalse(list(Path(folder).glob("*")))

    def test_malformed_pdf_and_failed_ingestion_are_not_searchable(self):
        with tempfile.TemporaryDirectory() as folder:
            self.main.BUNKER_DIR = Path(folder)
            with patch.object(self.main, "pdf_to_text", side_effect=DocumentValidationError("malformed")):
                with self.assertRaises(Exception) as caught:
                    self.call_upload(b"%PDF-malformed")
            self.assertEqual(getattr(caught.exception, "status_code", None), 400)
            self.assertEqual(json.loads(next(Path(folder).glob("*.json")).read_text())["status"], "failed")
        with tempfile.TemporaryDirectory() as folder:
            self.main.BUNKER_DIR = Path(folder)
            with patch.object(self.main, "pdf_to_text", return_value="policy text"), \
                 patch.object(self.main, "make_okf_markdown", return_value="okf"), \
                 patch.object(self.main, "ingest_markdown", side_effect=RuntimeError("secret must not leak")), \
                 patch.object(self.main, "get_qdrant", return_value=object()), \
                 patch.object(self.main, "get_embedding_model", return_value=object()):
                with self.assertRaises(Exception) as caught:
                    self.call_upload(b"%PDF-valid")
            self.assertEqual(getattr(caught.exception, "status_code", None), 503)
            self.assertNotIn("secret", str(caught.exception.detail))
            self.assertEqual(json.loads(next(Path(folder).glob("*.json")).read_text())["status"], "failed")

    def test_real_pdf_extractor_rejects_malformed_input(self):
        from document_pipeline import pdf_to_text
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.pdf"
            path.write_bytes(b"%PDF-not a valid document")
            with self.assertRaises(ValueError):
                pdf_to_text(path)

    def test_bad_extension_and_oversize_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            self.main.BUNKER_DIR = Path(folder)
            with self.assertRaises(Exception) as caught:
                self.call_upload(b"not pdf", filename="payload.exe")
            self.assertEqual(getattr(caught.exception, "status_code", None), 400)
            self.main.MAX_DOCUMENT_SIZE = 3
            try:
                with self.assertRaises(Exception) as caught:
                    self.call_upload(b"%PDF-too-large")
                self.assertEqual(getattr(caught.exception, "status_code", None), 413)
            finally:
                self.main.MAX_DOCUMENT_SIZE = 20 * 1024 * 1024

    def test_standalone_search_requires_a_role_and_uses_shared_filter(self):
        import search
        qdrant = FakeQdrant()
        model_module = SimpleNamespace(SentenceTransformer=lambda _: Model())
        with patch.object(search.argparse.ArgumentParser, "parse_args",
                          return_value=SimpleNamespace(question="question", role="intern")), \
             patch.object(search, "load_dotenv"), \
             patch.object(search.os, "getenv", side_effect=lambda key: "configured"), \
             patch.object(search, "QdrantClient", return_value=qdrant), \
             patch.dict(sys.modules, {"sentence_transformers": model_module}):
            search.main()
        self.assertEqual(qdrant.filters[0].must[0].match.any, ["intern"])

    def test_standalone_search_has_no_unrestricted_default_role(self):
        import search
        with patch.object(sys, "argv", ["search.py"]):
            with self.assertRaises(SystemExit) as caught:
                search.main()
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
