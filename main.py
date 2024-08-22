from fastapi import FastAPI

from model import NMT
from tokenizer import en_vec, ger_vec, ger_word_to_idx
from utils import get_model_metadata, is_model_exist

# Default model args
model_name, metadata = get_model_metadata("v8")

app = FastAPI()
model = NMT(en_vec,
            ger_vec,
            metadata["embedding_size"],
            metadata["hidden_units"])
model.translate("lorem ispum", ger_word_to_idx)
model.load_weights(metadata["model_path"])


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


@app.post("/model")
async def select_and_load_model(name: str):
    global model, model_name, metadata

    if name == model_name:
        pass
    else:
        if is_model_exist(name):
            model_name, metadata = get_model_metadata(name)
            model = NMT(en_vec,
                        ger_vec,
                        metadata["embedding_size"],
                        metadata["hidden_units"])

            model.translate("lorem ispum", ger_word_to_idx)
            model.load_weights(metadata["model_path"])

            return {
                "status": 200,
                "message": "Model loaded successfully!"
            }
        else:
            return {
                "status": 400,
                "message": "Model does not exist!"
            }
