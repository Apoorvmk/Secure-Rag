import os
import bcrypt

from datetime import datetime, timedelta

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from jose import jwt
from groq import Groq

from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from models import Base, User, Conversation, Message
from qdrant_client.models import PayloadSchemaType



# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
JWT_SECRET = os.getenv("JWT_SECRET")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

ALGORITHM = "HS256"
TOP_K = 3

ROLE_ACCESS = {
    "intern": ["intern"],
    "employee": ["intern", "employee"],
    "manager": ["intern", "employee", "manager"],
    "board": ["intern", "employee", "manager", "board"]
}

# Database
engine = create_engine(DATABASE_URL)

Base.metadata.create_all(bind=engine)


# FastAPI
app = FastAPI(title="Secure RAG")


# Groq
client = Groq(api_key=GROQ_API_KEY)


# JWT security
security = HTTPBearer()

# Qdrant 


QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)



if not qdrant.collection_exists("company_documents"):
    qdrant.create_collection(
        collection_name="company_documents",
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )

qdrant.create_payload_index(
    collection_name="company_documents",
    field_name="access_level",
    field_schema=PayloadSchemaType.KEYWORD
)
# -------------------------
# Authentication
# -------------------------

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    token = credentials.credentials

    try:
        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[ALGORITHM]
        )

        return payload

    except Exception:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token"
        )

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# -------------------------
# Basic endpoints
# -------------------------

@app.get("/")
def root():
    return {"message": "Secure RAG backend running"}


@app.get("/db-test")
def db_test():
    with engine.connect():
        return {"message": "Database connected successfully"}


# -------------------------
# Login
# -------------------------

@app.post("/login")
def login(email: str, password: str):

    with Session(engine) as session:

        user = session.query(User).filter(
            User.email == email
        ).first()

        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid credentials"
            )

        # Verify bcrypt password
        if not bcrypt.checkpw(
            password.encode("utf-8"),
            user.password_hash.encode("utf-8")
        ):
            raise HTTPException(
                status_code=401,
                detail="Invalid credentials"
            )

        token = jwt.encode(
            {
                "user_id": user.id,
                "role": user.role,
                "exp": datetime.utcnow() + timedelta(hours=2)
            },
            JWT_SECRET,
            algorithm=ALGORITHM
        )

        return {
            "access_token": token,
            "token_type": "bearer"
        }


# -------------------------
# Chat
# -------------------------
@app.post("/chat")
def chat(
    message: str,
    current_user: dict = Depends(get_current_user)
):
    user_id = current_user["user_id"]

    # Determine what this user is allowed to access
    user_role = current_user["role"]
    allowed_levels = ROLE_ACCESS[user_role]

    with Session(engine) as session:

        # 1. Get or create conversation
        conversation = session.query(Conversation).filter(
            Conversation.user_id == user_id
        ).order_by(Conversation.id.desc()).first()

        if not conversation:
            conversation = Conversation(user_id=user_id)
            session.add(conversation)
            session.commit()
            session.refresh(conversation)

        # 2. Save user's message
        user_message = Message(
            conversation_id=conversation.id,
            role="user",
            content=message
        )

        session.add(user_message)
        session.commit()

        # 3. Get conversation history
        messages = session.query(Message).filter(
            Message.conversation_id == conversation.id
        ).order_by(Message.id).all()

        chat_history = [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in messages
        ]

        # 4. Convert question into embedding
        query_vector = embedding_model.encode(message).tolist()

        # 5. Search ONLY authorized chunks
        results = qdrant.query_points(
            collection_name="company_documents",
            query=query_vector,
            query_filter={
                "must": [
                    {
                        "key": "access_level",
                        "match": {
                            "any": allowed_levels
                        }
                    }
                ]
            },
            limit=TOP_K
        ).points

        # 6. Build context from retrieved chunks
        context = "\n\n".join(
            result.payload["text"]
            for result in results
        )

        # 7. Give only authorized context to Groq
        rag_messages = [
            {
                "role": "system",
                "content": (
                    "Answer the user's question using the provided "
                    "company documents.\n\n"
                    "If the answer is not present in the provided "
                    "documents, say you don't know.\n\n"
                    f"Company documents:\n{context}"
                )
            }
        ]

        rag_messages.extend(chat_history)

        # 8. Ask Groq
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=rag_messages
        )

        answer = response.choices[0].message.content

        # 9. Save assistant response
        assistant_message = Message(
            conversation_id=conversation.id,
            role="assistant",
            content=answer
        )

        session.add(assistant_message)
        session.commit()

        # 10. Return response
        return {
            "user_id": user_id,
            "conversation_id": conversation.id,
            "response": answer
        }