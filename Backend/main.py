import os
import bcrypt
import mimetypes

from datetime import datetime, timedelta, timezone
from functools import lru_cache

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from jose import jwt
from groq import Groq

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from models import Base, User, Conversation, Message
from qdrant_client.models import PayloadSchemaType

import json
import uuid
from pathlib import Path

from fastapi import UploadFile, File, Form



# Load environment variables
load_dotenv(Path(__file__).with_name(".env"))

from access_policy import ROLES, TOP_K, allowed_access_levels, can_upload_to_level
from document_pipeline import (DocumentValidationError, filter_authorized_results,
                               ingest_markdown, make_okf_markdown, pdf_to_text,
                               retrieve_authorized)

ALGORITHM = "HS256"


class ConfigurationError(RuntimeError):
    """A safe, actionable configuration error with no credential values."""


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ConfigurationError(f"Missing required environment variable: {name}")
    return value


def _write_status(path: Path, value: dict) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary_path.replace(path)


@lru_cache(maxsize=1)
def get_engine():
    engine = create_engine(_required_env("DATABASE_URL"), pool_pre_ping=True)
    Base.metadata.create_all(bind=engine)
    return engine


# FastAPI
app = FastAPI(title="Secure RAG")
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "Frontend"


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=256)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


# JWT security
security = HTTPBearer()

@lru_cache(maxsize=1)
def get_qdrant():
    client = QdrantClient(url=_required_env("QDRANT_URL"), api_key=_required_env("QDRANT_API_KEY"))
    if not client.collection_exists("company_documents"):
        client.create_collection(collection_name="company_documents",
            vectors_config=VectorParams(size=384, distance=Distance.COSINE))
    client.create_payload_index(collection_name="company_documents", field_name="access_level",
        field_schema=PayloadSchemaType.KEYWORD)
    client.create_payload_index(collection_name="company_documents", field_name="document_id",
        field_schema=PayloadSchemaType.KEYWORD)
    return client


@lru_cache(maxsize=1)
def get_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def get_groq_client():
    return Groq(api_key=_required_env("GROQ_API_KEY"))
# -------------------------
# Authentication
# -------------------------

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    token = credentials.credentials

    try:
        jwt_secret = _required_env("JWT_SECRET")
    except ConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    try:
        payload = jwt.decode(token, jwt_secret, algorithms=[ALGORITHM])
        user_id = payload.get("user_id")
        if type(user_id) is not int:
            raise ValueError("Invalid token subject")
    except Exception:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token"
        )
    try:
        with Session(get_engine()) as session:
            user = session.query(User).filter(User.id == user_id).first()
            if not user or not user.is_active or not isinstance(user.role, str) or user.role not in ROLES:
                raise HTTPException(status_code=401, detail="Invalid or expired token")
            # Role always comes from the database, never from potentially stale JWT claims.
            return {"user_id": user.id, "role": user.role, "name": user.name}
    except HTTPException:
        raise
    except ConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Authentication service unavailable") from exc

# -------------------------
# Basic endpoints
# -------------------------

@app.get("/")
def root():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/db-test")
def db_test():
    try:
        with get_engine().connect():
            pass
    except ConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Database unavailable or misconfigured") from exc
    else:
        return {"message": "Database connected successfully"}


# -------------------------
# Login
# -------------------------

@app.post("/login")
def login(request: LoginRequest):

    try:
        with Session(get_engine()) as session:

            user = session.query(User).filter(
                User.email == request.email
            ).first()

            if not user or not user.is_active:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid credentials"
                )
            if not isinstance(user.role, str) or user.role not in ROLES:
                raise HTTPException(status_code=401, detail="Invalid credentials")

            # Verify bcrypt password
            if not bcrypt.checkpw(
                request.password.encode("utf-8"),
                user.password_hash.encode("utf-8")
            ):
                raise HTTPException(
                    status_code=401,
                    detail="Invalid credentials"
                )

            token = jwt.encode(
                {
                    "user_id": user.id,
                    "exp": datetime.now(timezone.utc) + timedelta(hours=2)
                },
                _required_env("JWT_SECRET"),
                algorithm=ALGORITHM
            )

            return {
                "access_token": token,
                "token_type": "bearer",
                "name": user.name,
                "role": user.role,
            }
    except HTTPException:
        raise
    except ConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Login service unavailable or misconfigured") from exc


@app.get("/me")
def who_am_i(current_user: dict = Depends(get_current_user)):
    return current_user


# -------------------------
# Chat
# -------------------------
@app.post("/chat")
def chat(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user)
):
    message = request.message
    user_id = current_user["user_id"]

    user_role = current_user["role"]
    try:
        results = retrieve_authorized(message, role=user_role, qdrant=get_qdrant(),
            model=get_embedding_model(), limit=TOP_K)
        # Missing/malformed access metadata is dropped defensively.
        safe_results = filter_authorized_results(results, user_role)
        context = "\n\n".join(point.payload["text"] for point in safe_results)
        with Session(get_engine()) as session:
            conversation = session.query(Conversation).filter(
                Conversation.user_id == user_id
            ).order_by(Conversation.id.desc()).first()
            if not conversation:
                conversation = Conversation(user_id=user_id)
                session.add(conversation)
                session.commit()
                session.refresh(conversation)
            session.add(Message(conversation_id=conversation.id, role="user", content=message))
            session.commit()
            # Reuse only user turns. Old assistant turns may contain material
            # retrieved under a more privileged role before a role downgrade.
            history = session.query(Message).filter(
                Message.conversation_id == conversation.id, Message.role == "user"
            ).order_by(Message.id).all()
            rag_messages = [{"role": "system", "content": (
                "Answer using the authorized company documents below. If absent, say you don't know.\n\n"
                f"Company documents:\n{context}")}]
            rag_messages.extend({"role": "user", "content": item.content} for item in history[-10:])
            response = get_groq_client().chat.completions.create(
                model="openai/gpt-oss-20b", messages=rag_messages)
            answer = response.choices[0].message.content
            if not isinstance(answer, str):
                raise RuntimeError("LLM returned an empty response")
            session.add(Message(conversation_id=conversation.id, role="assistant", content=answer))
            session.commit()
            return {"user_id": user_id, "conversation_id": conversation.id, "response": answer}
    except HTTPException:
        raise
    except ConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Chat service unavailable or misconfigured") from exc

BUNKER_DIR = Path(__file__).resolve().parent / "uploads"
MAX_DOCUMENT_SIZE = 20 * 1024 * 1024
MAX_VIDEO_SIZE = 100 * 1024 * 1024
FILE_TYPES = {
    ".pdf": "document", ".txt": "document", ".md": "document",
    ".png": "image", ".jpg": "image", ".jpeg": "image",
    ".gif": "image", ".webp": "image",
    ".mp4": "video", ".mov": "video", ".webm": "video",
}


class NoteRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    text: str = Field(min_length=1, max_length=100_000)
    access_level: str


def _validate_file_header(extension: str, category: str, header: bytes) -> None:
    valid = True
    if extension == ".pdf":
        valid = header.startswith(b"%PDF-")
    elif extension == ".png":
        valid = header.startswith(b"\x89PNG\r\n\x1a\n")
    elif extension in {".jpg", ".jpeg"}:
        valid = header.startswith(b"\xff\xd8\xff")
    elif extension == ".gif":
        valid = header.startswith((b"GIF87a", b"GIF89a"))
    elif extension == ".webp":
        valid = header.startswith(b"RIFF") and header[8:12] == b"WEBP"
    elif extension in {".mp4", ".mov"}:
        valid = len(header) >= 8 and header[4:8] == b"ftyp"
    elif extension == ".webm":
        valid = header.startswith(b"\x1a\x45\xdf\xa3")
    if not valid or not category:
        raise DocumentValidationError("File contents do not match the selected file type")


def _read_metadata(document_id: str) -> dict | None:
    try:
        parsed_id = str(uuid.UUID(document_id))
    except (ValueError, TypeError, AttributeError):
        return None
    path = BUNKER_DIR / f"{parsed_id}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) and data.get("document_id") == parsed_id else None


@app.post("/bunker/upload")
async def upload_document(
    file: UploadFile = File(...),
    access_level: str = Form(...),
    current_user: dict = Depends(get_current_user)
):
    if not can_upload_to_level(current_user.get("role"), access_level):
        await file.close()
        raise HTTPException(
            status_code=403,
            detail="You are not allowed to assign this access level"
        )

    if not file.filename:
        await file.close()
        raise HTTPException(status_code=400, detail="A file name is required")
    original_filename = file.filename.replace("\\", "/").rsplit("/", 1)[-1]
    original_filename = "".join(char for char in original_filename if char.isprintable()).strip()[:180]
    if not original_filename:
        await file.close()
        raise HTTPException(status_code=400, detail="A valid file name is required")
    extension = Path(original_filename).suffix.lower()
    category = FILE_TYPES.get(extension)
    if not category:
        await file.close()
        raise HTTPException(status_code=400,
            detail="Supported files: PDF, TXT, Markdown, common images, MP4, MOV, and WebM")

    document_id = str(uuid.uuid4())
    document_path = BUNKER_DIR / f"{document_id}{extension}"
    okf_path = BUNKER_DIR / f"{document_id}.md"
    metadata_path = BUNKER_DIR / f"{document_id}.json"
    max_size = MAX_VIDEO_SIZE if category == "video" else MAX_DOCUMENT_SIZE

    metadata = {
        "document_id": document_id,
        "original_filename": original_filename,
        "stored_extension": extension,
        "category": category,
        "title": Path(original_filename).stem,
        "access_level": access_level,
        "uploaded_by": current_user.get("user_id"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "processing"
    }

    vectors_written = False
    try:
        BUNKER_DIR.mkdir(parents=True, exist_ok=True)
        _write_status(metadata_path, metadata)
        total_size = 0
        header = bytearray()
        with document_path.open("wb") as destination:
            while chunk := await file.read(1024 * 1024):
                total_size += len(chunk)
                if total_size > max_size:
                    raise HTTPException(status_code=413,
                        detail=f"File exceeds the {max_size // (1024 * 1024)} MiB limit")
                if len(header) < 16:
                    header.extend(chunk[:16 - len(header)])
                destination.write(chunk)
        if total_size == 0:
            raise DocumentValidationError("The uploaded file is empty")
        _validate_file_header(extension, category, bytes(header))
        metadata["size_bytes"] = total_size
        if category == "document":
            if extension == ".pdf":
                text = pdf_to_text(document_path)
            else:
                try:
                    text = document_path.read_text(encoding="utf-8").strip()
                except UnicodeDecodeError as exc:
                    raise DocumentValidationError("Text documents must use UTF-8 encoding") from exc
                if not text:
                    raise DocumentValidationError("The text document is empty")
            markdown = make_okf_markdown(text, title=metadata["title"],
                access_level=access_level, source=original_filename,
                document_id=document_id, uploaded_by=current_user["user_id"])
            okf_path.write_text(markdown, encoding="utf-8")
            count = ingest_markdown(markdown, qdrant=get_qdrant(), model=get_embedding_model())
            vectors_written = True
            metadata.update(status="searchable", searchable=True, chunks=count)
        else:
            metadata.update(status="stored", searchable=False)
        _write_status(metadata_path, metadata)
    except HTTPException:
        metadata["status"] = "failed"
        if document_path.exists():
            try:
                document_path.unlink()
            except OSError:
                pass
        try:
            _write_status(metadata_path, metadata)
        except OSError:
            pass
        raise
    except DocumentValidationError as exc:
        metadata["status"] = "failed"
        try:
            _write_status(metadata_path, metadata)
        except OSError:
            pass
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        metadata["status"] = "failed"
        if vectors_written:
            try:
                qdrant = get_qdrant()
                from qdrant_client.models import FieldCondition, Filter, MatchValue
                qdrant.delete(collection_name="company_documents", points_selector=Filter(
                    must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]), wait=True)
            except Exception:
                pass
        try:
            _write_status(metadata_path, metadata)
        except OSError:
            pass
        detail = str(exc) if isinstance(exc, ConfigurationError) else "Document processing failed; it was not marked searchable"
        raise HTTPException(status_code=503, detail=detail) from exc
    finally:
        await file.close()

    return {
        "message": "Document stored successfully",
        "document_id": document_id,
        "access_level": access_level,
        "category": category,
        "searchable": metadata.get("searchable", False),
        "status": metadata["status"]
    }


@app.post("/bunker/notes")
def create_bunker_note(request: NoteRequest, current_user: dict = Depends(get_current_user)):
    if not can_upload_to_level(current_user.get("role"), request.access_level):
        raise HTTPException(status_code=403, detail="You are not allowed to assign this access level")
    title = request.title.strip()
    text = request.text.strip()
    if not title or not text:
        raise HTTPException(status_code=400, detail="A title and notice text are required")
    document_id = str(uuid.uuid4())
    filename = f"{title}.md"
    metadata = {"document_id": document_id, "original_filename": filename,
        "stored_extension": ".md", "category": "document", "title": title,
        "access_level": request.access_level, "uploaded_by": current_user["user_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "processing", "searchable": False}
    status_path = BUNKER_DIR / f"{document_id}.json"
    markdown_path = BUNKER_DIR / f"{document_id}.md"
    vectors_written = False
    try:
        BUNKER_DIR.mkdir(parents=True, exist_ok=True)
        _write_status(status_path, metadata)
        markdown = make_okf_markdown(text, title=title, access_level=request.access_level,
            source=filename, document_id=document_id, uploaded_by=current_user["user_id"])
        markdown_path.write_text(markdown, encoding="utf-8")
        chunks = ingest_markdown(markdown, qdrant=get_qdrant(), model=get_embedding_model())
        vectors_written = True
        metadata.update(status="searchable", searchable=True, chunks=chunks)
        _write_status(status_path, metadata)
        return {"document_id": document_id, "status": "searchable", "searchable": True}
    except Exception as exc:
        metadata.update(status="failed", searchable=False)
        if vectors_written:
            try:
                from qdrant_client.models import FieldCondition, Filter, MatchValue
                get_qdrant().delete(collection_name="company_documents", points_selector=Filter(
                    must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]), wait=True)
            except Exception:
                pass
        try:
            _write_status(status_path, metadata)
        except OSError:
            pass
        if isinstance(exc, HTTPException):
            raise
        if isinstance(exc, DocumentValidationError):
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        detail = str(exc) if isinstance(exc, ConfigurationError) else "Notice processing failed; it was not marked searchable"
        raise HTTPException(status_code=503, detail=detail) from exc


@app.get("/bunker")
def list_bunker_documents(current_user: dict = Depends(get_current_user)):
    permitted = set(allowed_access_levels(current_user["role"]))
    entries = []
    if not BUNKER_DIR.exists():
        return {"documents": []}
    for status_path in BUNKER_DIR.glob("*.json"):
        try:
            item = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (not isinstance(item, dict)
                or not isinstance(item.get("access_level"), str)
                or item["access_level"] not in permitted):
            continue
        entries.append({key: item.get(key) for key in (
            "document_id", "original_filename", "title", "category", "access_level",
            "status", "searchable", "size_bytes", "chunks", "uploaded_by", "created_at")})
    entries.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    return {"documents": entries}


@app.get("/bunker/files/{document_id}")
def get_bunker_file(document_id: str, current_user: dict = Depends(get_current_user)):
    metadata = _read_metadata(document_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Document not found")
    if (not isinstance(metadata.get("access_level"), str)
            or metadata["access_level"] not in allowed_access_levels(current_user["role"])):
        raise HTTPException(status_code=403, detail="You cannot access this document")
    if metadata.get("status") not in {"stored", "searchable"}:
        raise HTTPException(status_code=404, detail="Document file not found")
    extension = metadata.get("stored_extension")
    if extension not in FILE_TYPES:
        raise HTTPException(status_code=404, detail="Document file not found")
    path = BUNKER_DIR / f"{document_id}{extension}"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Document file not found")
    media_type = mimetypes.guess_type(metadata.get("original_filename", ""))[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type,
        filename=metadata.get("original_filename", f"document{extension}"))


app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="frontend-assets")
