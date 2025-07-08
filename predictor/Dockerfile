# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependency file and install the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the predictor script into the container
COPY main.py .

# Set the entrypoint to execute the predictor script
ENTRYPOINT ["python3", "main.py"]