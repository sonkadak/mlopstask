import os
import time
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
import json
import logging

# logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler("/tmp/triton_client.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Get env
TRITON_URL = os.getenv('TRITON_URL', 'localhost:8001')
MODEL_NAME = os.getenv('MODEL_NAME', 'your_model_name')

# triton client
try:
    triton_client = grpcclient.InferenceServerClient(url=TRITON_URL)
except Exception as e:
    logger.error(json.dumps({
        "event": "client_creation_failed",
        "error": str(e),
        "timestamp": time.time()
    }))
    exit(1)

# triton server check
if not triton_client.is_server_live():
    logger.error(json.dumps({
        "event": "server_not_live",
        "timestamp": time.time()
    }))
    exit(1)

if not triton_client.is_model_ready(MODEL_NAME):
    logger.error(json.dumps({
        "event": "model_not_ready",
        "model": MODEL_NAME,
        "timestamp": time.time()
    }))
    exit(1)

# input data
input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)

# Inference
inputs = grpcclient.InferInput('input', input_data.shape, 'FP32')
inputs.set_data_from_numpy(input_data)
outputs = grpcclient.InferRequestedOutput('output')

# Inference timing
while True:
    start_time = time.time()
    try:
        results = triton_client.infer(model_name=MODEL_NAME, inputs=[inputs], outputs=[outputs])
    except InferenceServerException as e:
        logger.error(json.dumps({
            "event": "inference_failed",
            "error": str(e),
            "timestamp": time.time()
        }))
        exit(1)
    end_time = time.time()
    
    # Inference latency
    inference_latency = (end_time - start_time) * 1000
    
    # Result data
    output_data = results.as_numpy('output')
    
    # Logging
    log_data = {
        "event": "inference_completed",
        "model": MODEL_NAME,
        "latency_ms": round(inference_latency, 2),
        "timestamp": time.time()
    }
    
    if output_data is not None:
        sampled_output = np.random.choice(output_data.flatten(), 10, replace=False).tolist()
        log_data["sampled_output"] = sampled_output
    else:
        log_data["status"] = "no_output_received"
    
    logger.info(json.dumps(log_data))
    time.sleep(10)
