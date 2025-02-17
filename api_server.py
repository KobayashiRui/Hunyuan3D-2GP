import argparse
import base64
import threading
import uuid
import logging
import os

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse


def build_logger(logger_name):
    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    return logger

SAVE_DIR = 'gradio_cache'
os.makedirs(SAVE_DIR, exist_ok=True)

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("controller", f"{SAVE_DIR}/controller.log")

class ModelWorker:
    def __init__(self, model_path="tencent/Hunyuan3D-2", profile="3"):
        app.state.logger.info(f"Loading the model {model_path} on worker ...")

    def generate(self, uid, params):
        if 'image' in params:
            image = params["image"]
        print(f"Worker {uid}")

        return uid



app = FastAPI()
@app.get("/status/ping")
async def root():
    return {"message": "Hello World"}

@app.post("/send")
async def generate(request: Request):
    logger.info("Worker send...")
    params = await request.json()
    uid = uuid.uuid4()
    threading.Thread(target=worker.generate, args=(uid, params,)).start()
    ret = {"uid": str(uid)}
    return JSONResponse(ret, status_code=200)

@app.get("/status/{uid}")
async def status(uid: str):
    save_file_path = os.path.join(SAVE_DIR, f'{uid}.glb')
    print(save_file_path, os.path.exists(save_file_path))
    if not os.path.exists(save_file_path):
        response = {'status': 'processing'}
        return JSONResponse(response, status_code=200)
    else:
        base64_str = base64.b64encode(open(save_file_path, 'rb').read()).decode()
        response = {'status': 'completed', 'model_base64': base64_str}
        return JSONResponse(response, status_code=200)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument('--profile', type=str, default="3")
    parser.add_argument('--verbose', type=str, default="1")

    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)

    worker = ModelWorker("tencent/Hunyuan3D-2", args.profile)
    app.state.logger = build_logger("controller", f"{SAVE_DIR}/controller.log")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")