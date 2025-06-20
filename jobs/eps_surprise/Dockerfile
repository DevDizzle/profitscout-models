FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# bring in shared helpers
COPY ../../utils ../utils
COPY . .

ENV PYTHONUNBUFFERED=1
CMD ["gunicorn", "-b", "0.0.0.0:8080", "main:app"]
