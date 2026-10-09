FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    FLASK_APP=app.py

WORKDIR /app

# Install dependencies first so this layer is cached
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code (dataset/artifacts are excluded via .dockerignore)
COPY . .

EXPOSE 5000

# app.py binds to 127.0.0.1 by default, so launch through the Flask CLI
# with 0.0.0.0 so the container port is reachable from the host.
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
