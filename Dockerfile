FROM python:3.12-slim

WORKDIR /app

# System deps for sqlite-vec and sentence-transformers
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY agent_memory/ agent_memory/
COPY scripts/ scripts/

# Data directory (mount as volume for persistence)
RUN mkdir -p /data
ENV AGENT_MEMORY_DATA_DIR=/data
ENV AGENT_MEMORY_DB=/data/memory.sqlite
ENV AGENT_MEMORY_HOST=0.0.0.0
ENV AGENT_MEMORY_PORT=8787

EXPOSE 8787

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -sf http://localhost:8787/v1/health || exit 1

CMD ["python", "-m", "agent_memory.api"]
