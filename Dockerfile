# BASIC IMAGE
FROM python:3.11-slim AS base

# SYSTEM PACKAGES
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
 && rm -rf /var/lib/apt/lists/*

# WORKING DIRECTORY
WORKDIR /app

# DEPENDENCIES
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# PROJECT CODE
COPY . /app

# PORTS
# Streamlit listens to port 8501 by default
EXPOSE 8501

# ЭНВИРОНМЕНТ
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CHROMA_TELEMETRY_ENABLED=false

# DEFAULT UI COMMANDS
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
