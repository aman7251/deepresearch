FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.huggingface

WORKDIR /app

# System deps for lxml/trafilatura
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8000 8501

# Default command runs the API; docker-compose overrides per service.
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
