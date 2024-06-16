# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the required files into the container
COPY src/train_pipeline.py /app
COPY src/predict.py /app
COPY src/config /app/config
COPY src/preprocessing /app/preprocessing
COPY src/pipeline.py /app

# Install any needed packages specified in requirements.txt
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/
# Run the training script to train the model when the container starts
CMD ["python", "train_pipeline.py"]
