"""Local structured progress; external dashboard integration is not included."""
import json


def update_model_status(model, **fields):
    print(json.dumps({'model': model, **fields}), flush=True)
