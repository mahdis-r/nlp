import os
import random
from pathlib import Path
from typing import List, Tuple, Optional

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


MODEL_FILENAME = "sentiment_model.joblib"


def build_dataset() -> Tuple[List[str], List[str]]:
    texts_good = [
        "I love this product so much",
        "Absolutely fantastic experience",
        "This is great and works perfectly",
        "What an amazing day",
        "The service was excellent and friendly",
        "I am very happy with the results",
        "Superb quality and great value",
        "Highly recommend to everyone",
        "This made me smile",
        "It exceeded my expectations",
        "Brilliant performance",
        "Top notch and delightful",
        "It was a pleasure to use",
        "Five stars, would buy again",
        "Wonderful and satisfying",
        "This is impressive work",
        "I enjoyed every moment",
        "It works like a charm",
        "Good job, well done",
        "I feel great about this",
    ]

    texts_bad = [
        "I hate this, it is terrible",
        "Absolutely awful experience",
        "This is bad and broken",
        "What a horrible day",
        "The service was rude and slow",
        "I am very disappointed",
        "Poor quality and a waste of money",
        "Would not recommend to anyone",
        "This made me angry",
        "It failed and wasted my time",
        "Terrible performance",
        "Bottom tier and frustrating",
        "It was painful to use",
        "One star, never again",
        "Awful and unsatisfying",
        "This is unimpressive work",
        "I regretted it immediately",
        "It barely works",
        "Bad job, not acceptable",
        "I feel terrible about this",
    ]

    texts_neutral = [
        "It is okay, nothing special",
        "The product is fine",
        "Average experience overall",
        "Not good, not bad",
        "It meets basic expectations",
        "This is acceptable",
        "Neutral feeling about this",
        "I neither like nor dislike it",
        "It works as expected",
        "Service was standard",
        "It is decent",
        "Nothing remarkable to mention",
        "It is just alright",
        "This is passable",
        "Reasonable quality",
        "Neither amazing nor awful",
        "Moderate performance",
        "It does the job",
        "I'm indifferent",
        "It's fine as is",
    ]

    X = texts_good + texts_bad + texts_neutral
    y = (["good"] * len(texts_good)) + (["bad"] * len(texts_bad)) + (["mutual"] * len(texts_neutral))

    paired = list(zip(X, y))
    random.shuffle(paired)
    X_shuffled, y_shuffled = zip(*paired)
    return list(X_shuffled), list(y_shuffled)


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    max_features=20000,
                    min_df=1,
                ),
            ),
            (
                "clf",
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
        ]
    )


def train_and_save(model_path: Optional[Path] = None) -> Path:
    if model_path is None:
        model_path = Path(__file__).with_name(MODEL_FILENAME)

    X, y = build_dataset()
    pipeline = build_pipeline()
    pipeline.fit(X, y)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Model trained and saved to {model_path}")
    return model_path


if __name__ == "__main__":
    train_and_save()