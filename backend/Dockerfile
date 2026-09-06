# Multi-stage build for security — minimal final image
FROM python:3.12-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-slim

# Security: non-root user
RUN groupadd -r drscreen && useradd -r -g drscreen -s /sbin/nologin drscreen

WORKDIR /app
COPY --from=builder /install /usr/local
COPY app/ ./app/

# OpenCV headless dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

USER drscreen

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
