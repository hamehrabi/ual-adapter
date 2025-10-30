# Multi-stage build for UAL Adapter

# Stage 1: Builder
FROM python:3.10-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
COPY pyproject.toml .
COPY setup.py .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder
COPY --from=builder /build/wheels /wheels

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --no-index --find-links=/wheels /wheels/* && \
    rm -rf /wheels

# Copy application code
COPY ual_adapter/ /app/ual_adapter/
COPY examples/ /app/examples/
COPY README.md /app/

# Create directories for models and outputs
RUN mkdir -p /app/models /app/adapters /app/outputs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/models
ENV UAL_ADAPTER_DIR=/app/adapters

# Create non-root user
RUN useradd -m -u 1000 ual_user && \
    chown -R ual_user:ual_user /app

USER ual_user

# Default command
CMD ["python", "-m", "ual_adapter.cli", "--help"]

# Labels
LABEL maintainer="your.email@example.com"
LABEL version="0.1.0"
LABEL description="Universal Adapter LoRA for architecture-agnostic model adaptation"
