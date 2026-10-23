"""Train the SentimentAI model.

Run from the project root:
    python model/train_model.py                  # uses model/train.csv (creates a DEMO one if missing)
    python model/train_model.py --data my.csv    # your own CSV with columns: text,label

Outputs (in model/):
    sentiment_model.joblib   - trained Logistic Regression
    tfidf_vectorizer.joblib  - TF-IDF vectorizer fitted on the TRAINING split only
    metrics.json             - evaluation results on the held-out test split
"""
import argparse
import json
import random
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split

# Make "app" importable no matter where the script is launched from
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.preprocessing import clean_text  # noqa: E402

MODEL_DIR = ROOT / "model"
DEFAULT_DATA = MODEL_DIR / "train.csv"
MODEL_PATH = MODEL_DIR / "sentiment_model.joblib"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
SEED = 42


# --------------------------------------------------------------------------
# 1. DEMO dataset (offline fallback). Synthetic - NOT real tweets/reviews.
# --------------------------------------------------------------------------
def build_demo_dataset(n_per_class: int = 600) -> pd.DataFrame:
    rng = random.Random(SEED)
    subjects = ["product", "movie", "service", "phone", "restaurant", "app", "hotel",
                "book", "game", "update", "laptop", "delivery", "course", "show", "store"]
    pos_adj = ["excellent", "fantastic", "amazing", "wonderful", "great", "brilliant",
               "superb", "lovely", "outstanding", "impressive", "delightful", "perfect"]
    neg_adj = ["terrible", "awful", "horrible", "disappointing", "poor", "dreadful",
               "useless", "pathetic", "annoying", "worthless", "unpleasant", "mediocre"]
    pos_templates = [
        "This {s} is {a}.", "I love this {s}, it is {a}!", "What a {a} {s}!",
        "The {s} was {a} and I would recommend it.", "Absolutely {a} {s}, totally worth it.",
        "I am so happy with the {s}, truly {a}.", "Not bad at all, the {s} is actually {a}.",
        "The {s} exceeded my expectations, {a} work.", "Best {s} I have had, {a} experience.",
        "Really enjoyed the {s}. It felt {a}.",
    ]
    neg_templates = [
        "This {s} is {a}.", "I hate this {s}, it is {a}!", "What a {a} {s}...",
        "The {s} was {a} and I would not recommend it.", "Absolutely {a} {s}, total waste of money.",
        "I am so unhappy with the {s}, truly {a}.", "Not good at all, the {s} is actually {a}.",
        "The {s} did not meet my expectations, {a} work.", "Worst {s} I have had, {a} experience.",
        "Really disliked the {s}. It felt {a}.",
    ]
    handwritten_pos = [
        "This product is excellent and I love it.", "Great customer support, they solved my issue quickly.",
        "I can't stop smiling, today was wonderful!", "Fast delivery and everything arrived perfectly.",
        "The team was friendly and very helpful.", "Totally worth the money, highly recommended.",
        "Such a beautiful day, I feel so grateful.", "Easy to use and works like a charm.",
        "I was pleasantly surprised by the quality.", "Five stars, I will definitely buy again.",
        "The food was delicious and the staff were kind.", "This made my whole week better.",
    ]
    handwritten_neg = [
        "This product is terrible and I regret buying it.", "Customer support ignored me for days.",
        "I can't believe how bad this was, so disappointing.", "Late delivery and the box was damaged.",
        "The staff were rude and unhelpful.", "Complete waste of money, do not buy.",
        "What a miserable day, everything went wrong.", "Hard to use and it keeps crashing.",
        "I was shocked by how poor the quality is.", "One star, I will never buy this again.",
        "The food was cold and the service was slow.", "This ruined my whole week.",
    ]

    def make(templates, adjectives, extras, label):
        texts = set(extras)
        while len(texts) < n_per_class:
            t = rng.choice(templates).format(s=rng.choice(subjects), a=rng.choice(adjectives))
            texts.add(t)
        return [(t, label) for t in texts]

    rows = make(pos_templates, pos_adj, handwritten_pos, "positive") + \
        make(neg_templates, neg_adj, handwritten_neg, "negative")
    rng.shuffle(rows)
    return pd.DataFrame(rows, columns=["text", "label"])


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[data] {path} not found -> creating the DEMO dataset (synthetic, for learning only).")
        path.parent.mkdir(parents=True, exist_ok=True)
        build_demo_dataset().to_csv(path, index=False)
    df = pd.read_csv(path)
    if not {"text", "label"}.issubset(df.columns):
        sys.exit("ERROR: the CSV must have the columns 'text' and 'label'.")
    df = df[["text", "label"]].dropna()
    # Accept 0/1 or 0/4 labels as well as words
    mapping = {0: "negative", 1: "positive", 4: "positive", "0": "negative", "1": "positive", "4": "positive"}
    df["label"] = df["label"].map(lambda v: mapping.get(v, str(v).strip().lower()))
    unknown = set(df["label"]) - {"positive", "negative"}
    if unknown:
        sys.exit(f"ERROR: unexpected labels {unknown}. Use positive/negative (or 0/1).")
    return df.drop_duplicates(subset="text").reset_index(drop=True)


# --------------------------------------------------------------------------
# 2. Train / evaluate / save
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SentimentAI model")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="CSV with columns text,label")
    args = parser.parse_args()

    df = load_dataset(args.data)
    print(f"[data] {len(df)} rows | class counts: {df['label'].value_counts().to_dict()}")

    df["clean"] = df["text"].map(clean_text)
    df = df[df["clean"] != ""]

    # Split FIRST, then fit the vectorizer on training data only (no leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"], test_size=0.2, random_state=SEED, stratify=df["label"])

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=50000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)           # transform only - never fit on test

    model = LogisticRegression(max_iter=1000, random_state=SEED)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    labels = list(model.classes_)
    metrics = {
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, pos_label="positive"), 4),
        "recall": round(recall_score(y_test, y_pred, pos_label="positive"), 4),
        "f1": round(f1_score(y_test, y_pred, pos_label="positive"), 4),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
    }
    print("\n=== Evaluation on the held-out TEST set ===")
    print(classification_report(y_test, y_pred, digits=3))
    print(f"Confusion matrix (rows=true, cols=predicted, order={labels}):")
    print(metrics["confusion_matrix"])

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"\n[saved] {MODEL_PATH.name}, {VECTORIZER_PATH.name}, {METRICS_PATH.name}")

    # Verify the saved files really work
    m = joblib.load(MODEL_PATH)
    v = joblib.load(VECTORIZER_PATH)
    for sentence in ["I love this, it is fantastic!", "This was awful and a waste of money."]:
        proba = m.predict_proba(v.transform([clean_text(sentence)]))[0]
        print(f"[verify] {sentence!r} -> {m.classes_[proba.argmax()]} ({proba.max():.2f})")

    if len(df) < 5000:
        print("\nNOTE: small dataset -> scores are optimistic and NOT production quality.")


if __name__ == "__main__":
    main()
