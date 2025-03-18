# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port Streamlit uses (default is 8501)
EXPOSE 8501

# Define the command to run your app using Streamlit
CMD ["streamlit", "run", "failure.py", "--server.enableCORS", "false", "--server.port", "8501"]

