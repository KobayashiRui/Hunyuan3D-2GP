import argparse
import uvicorn
from fastapi import FastAPI

app = FastAPI()

class ModelWorker:
    def __init__(self, model_path="tencent/Hunyuan3D-2", profile="3"):
        pass


@app.get("/")
async def root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument('--profile', type=str, default="3")
    parser.add_argument('--verbose', type=str, default="1")

    args = parser.parse_args()
    app.state.worker = ModelWorker("tencent/Hunyuan3D-2", args.profile)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")