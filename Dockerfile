FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir flask lightgbm numpy pandas

COPY serve.py .

CMD ["python", "serve.py"]
