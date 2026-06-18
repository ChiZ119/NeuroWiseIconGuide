FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git unzip \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /home/user/app

# Install dependencies first (faster rebuilds)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model into the image so the first
# request doesn't have to wait for it.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the rest of the project
COPY --chown=user . .
RUN unzip -q size-512-images.zip && rm size-512-images.zip

EXPOSE 7860

# Long timeout because loading torch + spaCy + embedding 8,003 images takes a while at boot.
CMD ["gunicorn", "serverOfNER:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "600"]