# Use the official Python base image
FROM python:3.9-slim

# Set environment variables to ensure Python behaves as expected
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files into the container
COPY . .

# Expose the application port
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "fastapi-app:app", "--host", "0.0.0.0", "--port", "8000"]
