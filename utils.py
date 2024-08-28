import json


def save_model_metadata(model_name,
                        model_path,
                        embedding_size,
                        hidden_units,
                        save_path="./weights.jsonl"):
    metadata = {
        "model_path": model_path,
        "embedding_size": embedding_size,
        "hidden_units": hidden_units,
    }

    with open(save_path, "w") as f:
        data = json.load(f)
        data[model_name] = metadata
        json.dump(data, f)


def get_model_metadata(model_name, save_path="./weights.jsonl"):
    with open(save_path, "r") as f:
        data = json.load(f)
        metadata = data[model_name]

    return model_name, metadata


def retrieve_all_models(path="./weights.jsonl"):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def is_model_exist(model_name, save_path="./weights.jsonl"):
    with open(save_path, "r") as f:
        data = json.load(f)
        if model_name in data:
            return True

    return False
