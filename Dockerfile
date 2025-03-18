FROM python:3.12-slim-buster

# Install necessary system packages
RUN apt update -y && apt install -y awscli

# Set the working directory
WORKDIR /app

# Copy all files into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make the startup script executable
RUN chmod +x start.sh

# Run both FastAPI/Flask app and Streamlit UI
CMD ["./start.sh"]