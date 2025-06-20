FROM python:3.10-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY utils ./utils
COPY . . 

RUN ls -l /app   
CMD ["gunicorn", "-b", "0.0.0.0:8080", "main:app"]
