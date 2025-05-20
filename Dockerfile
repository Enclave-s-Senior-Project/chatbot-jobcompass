FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y gcc build-essential libpq-dev && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirement.txt ./
RUN pip install --upgrade pip && pip install --prefix=/install -r requirement.txt && rm -rf ~/.cache/pip

# Copy only what's needed to run the NLTK download script
COPY app/utils ./app/utils
COPY scripts/download_nltk_data.py ./scripts/download_nltk_data.py
RUN PYTHONPATH="/install/lib/python3.12/site-packages:." python scripts/download_nltk_data.py

# --- Final image ---
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y libpq-dev && rm -rf /var/lib/apt/lists/* && pip install huggingface_hub[hf_xet]

# Copy installed python packages from builder
COPY --from=builder /install /usr/local

# Copy project files
COPY app ./app
COPY scripts ./scripts
COPY requirement.txt ./
COPY constants.py ./

# Ensure data directory exists
RUN mkdir -p /app/app/data

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]