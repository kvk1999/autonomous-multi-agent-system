# Use a slim Python 3.11 runtime (stable, smaller than 3.14 for Cloud Run)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching)
COPY requirements.txt .

# Install CPU-only PyTorch (avoids 3GB GPU binaries, Cloud Run doesn't have GPU)
# All other packages install normally
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir streamlit plotly pandas numpy google-genai \
    google-cloud-storage google-cloud-bigquery google-cloud-aiplatform \
    pyarrow Flask gunicorn

# Copy all application files
COPY . .

# Cloud Run injects PORT env variable; Streamlit listens on it
ENV PORT=8080

EXPOSE 8080

# Run Streamlit on the Cloud Run expected port
ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=8080", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--browser.gatherUsageStats=false"]