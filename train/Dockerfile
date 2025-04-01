FROM python:3.10-slim

ENV CONT_NAME="train/resnet18-mnist"
ENV CONT_TAG="0.4.3"

RUN apt-get update && apt-get install -y git \
    && pip install torch torchvision onnx onnxruntime

WORKDIR /app
COPY train.py .

ENTRYPOINT ["python", "train.py", "--debug"]
