import argparse
import asyncio
import base64
import logging
import logging.handlers
import os
import sys
import tempfile
import threading
import uuid
from io import BytesIO


import torch
import trimesh
import uvicorn
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FloaterRemover, DegenerateFaceRemover, FaceReducer
#from hy3dgen.texgen import Hunyuan3DPaintPipeline
#from hy3dgen.text2image import HunyuanDiTPipeline
from mmgp import offload

LOGDIR = "."
handler = None

def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when='D', utc=True, encoding='UTF-8')
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger

class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"



SAVE_DIR = 'gradio_cache'
os.makedirs(SAVE_DIR, exist_ok=True)

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("controller", f"{SAVE_DIR}/controller.log")

def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))

class ModelWorker:
    def __init__(self, model_path="tencent/Hunyuan3D-2"):
        self.model_path = model_path
        self.worker_id = str(uuid.uuid4())[:6]
        self.profile = int(args.profile)
        self.verbose = int(args.verbose)

        self.rembg_wroker = BackgroundRemover()
        self.i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(self.model_path, device="cpu", use_safetensors = True)
        self.floater_remove_worker = FloaterRemover()
        self.degenerate_face_remove_worker = DegenerateFaceRemover()
        self.face_reduce_worker = FaceReducer()
        self.kwargs = {}
        self.pipe = offload.extract_models("i23d_worker", self.i23d_worker)
        if self.profile < 5:
            self.kwargs["pinnedMemory"] = "i23d_worker/model"
        if self.profile !=1 and self.profile !=3:
            self.kwargs["budgets"] = { "*" : 2200 }
        offload.profile(self.pipe, profile_no = self.profile, verboseLevel = self.verbose, **self.kwargs)


    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return args.limit_model_concurrency - model_semaphore._value + (len(
                model_semaphore._waiters) if model_semaphore._waiters is not None else 0)

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return args.limit_model_concurrency - model_semaphore._value + (len(
                model_semaphore._waiters) if model_semaphore._waiters is not None else 0)

    def get_status(self):
        return {
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    @torch.inference_mode()
    def generate(self, uid, params):
        if 'image' in params:
            image = params["image"]
            image = load_image_from_base64(image)
        else:
            raise ValueError("No input image")
        
        image = self.rembg_wroker(image)
        params['image'] = image

        seed = params.get("seed", 1234)
        generator = torch.Generator()
        params['generator'] = generator.manual_seed(seed)
        params['octree_resolution'] = params.get("octree_resolution", 256)
        params['num_inference_steps'] = params.get("num_inference_steps", 30)
        params['guidance_scale'] = params.get('guidance_scale', 7.5)
        mesh = self.i23d_worker(**params)[0]

        mesh = FloaterRemover()(mesh)
        mesh = DegenerateFaceRemover()(mesh)
        mesh = FaceReducer()(mesh)

        with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as temp_file:
            mesh.export(temp_file.name)
            mesh = trimesh.load(temp_file.name)
            temp_file.close()
            os.unlink(temp_file.name)
            save_path = os.path.join(SAVE_DIR, f'{str(uid)}.glb')
            mesh.export(save_path)

        #torch.cuda.empty_cache()
        return save_path, uid



app = FastAPI()
@app.get("/status/ping")
async def root():
    return {"message": "Hello World"}

@app.get("/generate")
async def generate(request: Request):
    logger.info("Worker generate...")
    params = await request.json()
    uid = uuid.uuid4()
    try:
        file_path, uid = worker.generate(uid, params)
        return FileResponse(file_path)
    except Exception as e:
        logger.error("Caught Error", e)
        ret = {
            "text": "Server Error",
            "error_code": 1,
        }
        return JSONResponse(ret, status_code=404)
    

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
    parser.add_argument("--limit-model-concurrency", type=int, default=5)

    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)

    model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    worker = ModelWorker("tencent/Hunyuan3D-2")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")