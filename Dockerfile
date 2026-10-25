FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Train models at build time so the API can predict immediately
RUN python -c "from src.model import train_all; train_all()"
EXPOSE 8080
CMD ["python", "-m", "src.app"]
