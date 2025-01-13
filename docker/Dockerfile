# Use the Python 3.10 slim base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt ./requirements.txt

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY prod ./prod
COPY src ./src
COPY dataset ./dataset
COPY models ./models

# Set environment variable for Python to locate the src directory
ENV PYTHONPATH=/app

# Set the default command to run the inference script
ENTRYPOINT ["python", "prod/run_inference.py"]
