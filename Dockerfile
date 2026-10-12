FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 MLFLOW_DISABLE_AGENT_HINT=1 MLFLOW_DISABLE_TELEMETRY=true

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/

# Train inside the image so the model is built with the exact library
# versions that serve it (prevents train/serve version mismatch).
RUN python -m src.download_data && python -m src.train

EXPOSE 8000
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
