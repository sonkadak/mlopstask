FROM python:3.8-slim

ENV CONT_TAG="0.1.3"

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libssl-dev

RUN pip install --no-cache-dir \
    tritonclient[grpc] \
    numpy

COPY client.py .

CMD ["python", "client.py"]

