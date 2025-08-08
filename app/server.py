from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
from flask import Flask, jsonify, render_template, request

# Support both "python -m app.server" and "python app/server.py"
try:
    # When running as a module: py -m app.server
    from .train import train_and_save, MODEL_FILENAME
except Exception:
    # When running as a script: py app\server.py
    from train import train_and_save, MODEL_FILENAME  # type: ignore


app = Flask(__name__, template_folder="templates", static_folder="static")
MODEL_PATH = Path(__file__).with_name(MODEL_FILENAME)


def load_or_train_model():
    if not MODEL_PATH.exists():
        train_and_save(MODEL_PATH)
    return joblib.load(MODEL_PATH)


model = load_or_train_model()


@app.route("/", methods=["GET"])
def index() -> str:
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict() -> Any:
    try:
        payload: Dict[str, Any] = request.get_json(force=True, silent=False) or {}
    except Exception:
        payload = {}

    text = (payload.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Missing 'text' in JSON body"}), 400

    try:
        label = model.predict([text])[0]
        if label not in {"good", "bad", "mutual"}:
            label = "mutual"
        return jsonify({"label": label})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)