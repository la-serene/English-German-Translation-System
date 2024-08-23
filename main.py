import tensorflow as tf

from fastapi import FastAPI

from tokenizer import ger_word_to_idx
from utils import *

# Default model args
model_name, metadata = get_model_metadata("v9")

app = FastAPI()
model = tf.keras.models.load_model(metadata["model_path"])


@app.get("/")
async def root():
    return {
        "message": "Hello World!"
    }


@app.post("/translate")
async def translate(text: str):
    translation = model.translate(text, ger_word_to_idx)
    return {
        "translation": translation
    }


@app.get("/get_models")
async def get_models(path="./weights.jsonl"):
    data = retrieve_all_models(path)
    return data


@app.post("/model")
async def select_and_load_model(name: str):
    global model, model_name, metadata

    if name == model_name:
        pass
    else:
        if is_model_exist(name):
            _, metadata = get_model_metadata(name)
            model = tf.keras.models.load_model(metadata["model_path"])

            return {
                "status": 200,
                "message": "Model loaded successfully!"
            }
        else:
            return {
                "status": 400,
                "message": "Model does not exist!"
            }
