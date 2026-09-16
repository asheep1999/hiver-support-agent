import os
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support
)

from src.classifier import build_model
from src.retrieval import AppleSupportRetriever
from src.escalation import decide_escalation


DEV_PATH = "data/processed/apple_support_dev.csv"
GOLDEN_PATH = "data/golden/apple_support_golden_set.csv"
RESULTS_PATH = "results/agent_predictions.csv"


def train_classifier():
    dev_df = pd.read_csv(DEV_PATH)

    X_train = dev_df["clean_customer_message"].astype(str)
    y_train = dev_df["intent"].astype(str)

    model = build_model()
    model.fit(X_train, y_train)

    return model


def evaluate_agent():

    os.makedirs("results", exist_ok=True)

    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    golden = pd.read_csv(GOLDEN_PATH)

    print("Golden examples:", len(golden))

    # -----------------------------------------
    # Train intent classifier
    # -----------------------------------------

    print("\nTraining intent classifier...")

    classifier = train_classifier()

    # -----------------------------------------
    # Load semantic retriever
    # -----------------------------------------

    print("\nLoading semantic retriever...")

    retriever = AppleSupportRetriever()

    predicted_intents = []
    predicted_escalations = []
    confidences = []
    top_similarities = []
    escalation_reasons = []

    # -----------------------------------------
    # Evaluate all 200 examples
    # IMPORTANT:
    # No Gemini API is called here.
    # -----------------------------------------

    print("\nEvaluating Golden Set...\n")

    for i, row in golden.iterrows():

        message = str(row["clean_customer_message"])

        # 1. Intent
        predicted_intent = classifier.predict(
            [message]
        )[0]

        probabilities = classifier.predict_proba(
            [message]
        )[0]

        confidence = float(probabilities.max())

        # 2. Historical retrieval
        historical_cases = retriever.search(
            message,
            top_k=3
        )

        if historical_cases:
            top_similarity = historical_cases[0]["score"]
        else:
            top_similarity = 0.0

        # 3. Escalation
        escalation = decide_escalation(
            intent=predicted_intent,
            confidence=confidence,
            historical_cases=historical_cases
        )

        predicted_intents.append(predicted_intent)

        predicted_escalations.append(
            "yes" if escalation["escalate"] else "no"
        )

        confidences.append(confidence)

        top_similarities.append(top_similarity)

        escalation_reasons.append(
            escalation["reason"]
        )

        if (i + 1) % 20 == 0:
            print(
                f"Processed {i + 1}/{len(golden)}"
            )

    # -----------------------------------------
    # INTENT EVALUATION
    # -----------------------------------------

    y_true_intent = (
        golden["intent"].astype(str)
    )

    print("\n" + "=" * 70)
    print("INTENT CLASSIFICATION RESULTS")
    print("=" * 70)

    intent_accuracy = accuracy_score(
        y_true_intent,
        predicted_intents
    )

    print(
        "\nAccuracy:",
        round(intent_accuracy, 4)
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_true_intent,
            predicted_intents,
            zero_division=0
        )
    )

    # -----------------------------------------
    # ESCALATION EVALUATION
    # -----------------------------------------

    y_true_escalation = (
        golden["expected_escalation"]
        .astype(str)
    )

    print("\n" + "=" * 70)
    print("ESCALATION RESULTS")
    print("=" * 70)

    print(
        classification_report(
            y_true_escalation,
            predicted_escalations,
            labels=["yes", "no"],
            zero_division=0
        )
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true_escalation,
            predicted_escalations,
            pos_label="yes",
            average="binary",
            zero_division=0
        )
    )


    print(
        "Escalation Precision:",
        round(precision, 4)
    )

    print(
        "Escalation Recall:",
        round(recall, 4)
    )

    print(
        "Escalation F1:",
        round(f1, 4)
    )

    # -----------------------------------------
    # SAVE RESULTS
    # -----------------------------------------

    results = golden.copy()

    results["predicted_intent"] = predicted_intents

    results["predicted_escalation"] = (
        predicted_escalations
    )

    results["intent_confidence"] = confidences

    results["top_retrieval_similarity"] = (
        top_similarities
    )

    results["escalation_reason"] = (
        escalation_reasons
    )

    results.to_csv(
        RESULTS_PATH,
        index=False
    )

    print("\n" + "=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)

    print(RESULTS_PATH)


if __name__ == "__main__":
    evaluate_agent()