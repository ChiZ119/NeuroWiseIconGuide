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

# Pre-download the text and CLIP models into the image so the first
# request doesn't have to wait for them.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); SentenceTransformer('clip-ViT-B-32')"

# Copy the rest of the project
COPY --chown=user . .

# Download the ARASAAC pictures for the prebuilt index (items.json / index.npz
# are committed; only the image files are fetched here).
RUN python buildPictogramIndex.py images --library pictograms/arasaac-en

ENV PICTO_LIBRARY=pictograms/arasaac-en

EXPOSE 7860

# Long timeout because loading torch + spaCy + the picture index takes a while at boot.
CMD ["gunicorn", "serverOfNER:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "600"]
