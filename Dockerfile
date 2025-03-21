FROM python:3.12-slim

ENV MLFLOW_TRACKING_URI=https://dagshub.com/vivekprasanna.prabhu/ShopTalk.mlflow
ENV MODEL_DIR="/app/fine_tuned_lora_tripletloss"

# Install necessary system packages for wordcloud and other dependencies
RUN apt update -y && apt install -y \
    awscli \
    gcc \
    libpng-dev

# Set the working directory
WORKDIR /app

# Copy all files into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make the startup script executable
RUN chmod +x start.sh

EXPOSE 8080 8501

# Run both FastAPI/Flask app and Streamlit UI
CMD ["./start.sh"]
