FROM python:3.8-slim-buster

RUN apt update -y && apt install awscli -y
WORKDIR /app

COPY . /app
RUN pip install -r requirements.txt

RUN chmod +x start.sh  # Make the script executable
CMD ["./start.sh"]