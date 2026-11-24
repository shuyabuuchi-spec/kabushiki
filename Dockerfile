FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config/ ./config/
COPY kabushiki/ ./kabushiki/
COPY web/ ./web/

RUN mkdir -p /app/data /app/web/data

EXPOSE 8000

CMD ["uvicorn", "kabushiki.server:app", "--host", "0.0.0.0", "--port", "8000"]
