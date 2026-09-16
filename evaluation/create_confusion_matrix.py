import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


PREDICTIONS_PATH = "results/agent_predictions.csv"
OUTPUT_PATH = "results/confusion_matrix.png"


def main():
    if not os.path.exists(PREDICTIONS_PATH):
        raise FileNotFoundError(
            f"Could not find {PREDICTIONS_PATH}. "
            "Run the evaluation first."
        )

    df = pd.read_csv(PREDICTIONS_PATH)

    required_columns = ["intent", "predicted_intent"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Required column '{column}' is missing from {PREDICTIONS_PATH}"
            )

    labels = sorted(
        set(df["intent"].astype(str))
        | set(df["predicted_intent"].astype(str))
    )

    cm = confusion_matrix(
        df["intent"].astype(str),
        df["predicted_intent"].astype(str),
        labels=labels
    )

    fig, ax = plt.subplots(figsize=(12, 10))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    display.plot(
        ax=ax,
        xticks_rotation=45
    )

    ax.set_title("AppleSupport Intent Classification Confusion Matrix")
    ax.set_xlabel("Predicted Intent")
    ax.set_ylabel("Actual Intent")

    plt.tight_layout()

    os.makedirs("results", exist_ok=True)

    plt.savefig(
        OUTPUT_PATH,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved confusion matrix to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()