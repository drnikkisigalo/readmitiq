# ============================================================
# ReadmitIQ — Dockerfile
# Builds the FastAPI inference service into a container.
# The model artifact is loaded from AWS S3 at startup,
# not baked into the image. This keeps the image small and
# allows model updates without rebuilding.
#
# Build:  docker build -t readmitiq .
# Run:    docker run -p 8000:8000 --env-file .env readmitiq
# Test:   curl http://localhost:8000/health
# ============================================================

# Use a slim Python base image to keep the container small
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements first — Docker caches this layer.
# If requirements don't change, pip install is skipped on rebuild.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API and source code into the container
COPY api/ ./api/
COPY src/ ./src/

# Expose the port FastAPI will run on
EXPOSE 8000

# Health check — Docker will ping this to confirm the container is healthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start the FastAPI server
# --host 0.0.0.0 makes it accessible outside the container
# --workers 1 is appropriate for a single Fargate task
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
