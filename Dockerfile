FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY Backend/requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && python -m pip install -r /app/requirements.txt

COPY Backend /app/Backend
COPY Frontend /app/Frontend
COPY documents/okf /app/documents/okf
RUN mkdir -p /app/Backend/uploads

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "main:app", "--app-dir", "Backend", "--host", "0.0.0.0", "--port", "8000"]
