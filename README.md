# Sentiment Classifier (good / bad / mutual)

This project trains a simple text sentiment classifier (good, bad, mutual) with scikit-learn, exposes it via a Flask API, and serves a small web page to interact with it.

## 1) Setup (local Python)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Train the model

```bash
python app/train.py
```

This creates `app/sentiment_model.joblib`.

Note: If you skip this step, the server will train automatically on its first start.

## 3) Run the web app (local)

```bash
python app/server.py
```

Open http://localhost:8000 in your browser.

## 4) Docker (build and run)

```bash
docker build -t sentiment-app:latest .
docker run --rm -p 8000:8000 sentiment-app:latest
```

Open http://localhost:8000.

## API

- POST `/predict` with JSON: `{ "text": "..." }` -> `{ "label": "good|bad|mutual" }`

## Notes

- The included dataset is small and only for demonstration. For real applications, train with a larger, labeled dataset.
- The model uses TF-IDF + Logistic Regression. It is very fast and lightweight, ideal for simple demos.