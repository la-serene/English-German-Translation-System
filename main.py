import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
from model import NMT   # Locate model

from utils import *


class RequestModel(BaseModel):
    inputs: str
    maxlen: int = 40


# Default model args
model_name, metadata = get_model_metadata("v9")

app = FastAPI()
model = tf.keras.models.load_model(metadata["path"])


@app.get("/")
async def root():
    return {
        "message": "Hello World!"
    }


@app.post("/translate")
async def translate(data: RequestModel):
    inputs = data.inputs
    maxlen = data.maxlen
    translation = model.translate(next_inputs=inputs, maxlen=maxlen)
    return {
        "translation": translation
    }


@app.get("/get_models")
async def get_models(path="./weights.jsonl"):
    data = retrieve_all_models(path)
    return data


# @app.post("/model")
# async def select_and_load_model(name: str):
#     global model, model_name, metadata
#
#     if name == model_name:
#         pass
#     else:
#         if await is_model_exist(name):
#             _, metadata = await get_model_metadata(name)
#             model = await tf.keras.models.load_model(metadata["model_path"])
#
#             return {
#                 "status": 200,
#                 "message": "Model loaded successfully!"
#             }
#         else:
#             return {
#                 "status": 400,
#                 "message": "Model does not exist!"
#             }
