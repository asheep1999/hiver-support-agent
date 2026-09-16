import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report
)


GOLDEN_PATH = "data/golden/apple_support_golden_set.csv"


def majority_baseline():
    df = pd.read_csv(GOLDEN_PATH)

    y_true = df["intent"].astype(str)

    majority_intent = y_true.value_counts().idxmax()

    y_pred = [majority_intent] * len(y_true)

    print("=" * 60)
    print("Majority-Class Baseline")
    print("=" * 60)

    print("\nMajority intent:")
    print(majority_intent)

    print("\nAccuracy:")
    print(accuracy_score(y_true, y_pred))

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0
        )
    )


if __name__ == "__main__":
    majority_baseline()