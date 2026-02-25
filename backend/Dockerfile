FROM python:3.11-slim

WORKDIR /app

COPY backend/ /app/

RUN pip install --no-cache-dir -r requirements.txt

ENV PORT=7860
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}