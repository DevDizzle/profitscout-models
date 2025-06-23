# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code (main.py, bq.py) into the container
COPY . .

# Set the entrypoint to run your script directly with Python.
# This is the most important line - it bypasses the search for 'app'.
ENTRYPOINT ["python", "main.py"]