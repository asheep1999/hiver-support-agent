import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


DEV_PATH = "data/processed/apple_support_dev.csv"
GOLDEN_PATH = "data/golden/apple_support_golden_set.csv"


def load_data():
    dev_df = pd.read_csv(DEV_PATH)
    golden_df = pd.read_csv(GOLDEN_PATH)

    X_train = dev_df["clean_customer_message"].astype(str)
    y_train = dev_df["intent"].astype(str)

    X_test = golden_df["clean_customer_message"].astype(str)
    y_test = golden_df["intent"].astype(str)

    return X_train, y_train, X_test, y_test


def build_model():
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            )
        )
    ])


def train_and_evaluate():
    X_train, y_train, X_test, y_test = load_data()

    print("Training examples:", len(X_train))
    print("Golden test examples:", len(X_test))

    model = build_model()

    print("\nTraining model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\n" + "=" * 60)
    print("TF-IDF + Logistic Regression Results")
    print("=" * 60)

    print("\nAccuracy:")
    print(accuracy_score(y_test, predictions))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    labels = sorted(y_test.unique())

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    )

    print(cm_df)

    return model


if __name__ == "__main__":
    train_and_evaluate()