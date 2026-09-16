import pandas as pd
from collections import Counter


RESULTS_PATH = "results/agent_predictions.csv"


def analyze_failures():

    df = pd.read_csv(RESULTS_PATH)

    # ------------------------------------------------
    # 1. Intent failures
    # ------------------------------------------------

    intent_failures = df[
        df["intent"] != df["predicted_intent"]
    ].copy()

    print("=" * 70)
    print("INTENT FAILURE SUMMARY")
    print("=" * 70)

    print("Total intent failures:", len(intent_failures))
    print(
        "Intent failure rate:",
        round(len(intent_failures) / len(df), 4)
    )

    print("\nMost common actual → predicted confusions:")

    confusion_counts = (
        intent_failures
        .groupby(["intent", "predicted_intent"])
        .size()
        .sort_values(ascending=False)
    )

    print(confusion_counts.head(15))

    # ------------------------------------------------
    # 2. Escalation failures
    # ------------------------------------------------

    escalation_failures = df[
        df["expected_escalation"]
        != df["predicted_escalation"]
    ].copy()

    print("\n" + "=" * 70)
    print("ESCALATION FAILURE SUMMARY")
    print("=" * 70)

    print(
        "Total escalation failures:",
        len(escalation_failures)
    )

    # False positives:
    # predicted yes, actually no

    false_positives = df[
        (df["expected_escalation"] == "no")
        &
        (df["predicted_escalation"] == "yes")
    ].copy()

    # False negatives:
    # predicted no, actually yes

    false_negatives = df[
        (df["expected_escalation"] == "yes")
        &
        (df["predicted_escalation"] == "no")
    ].copy()

    print(
        "\nFalse positives:",
        len(false_positives)
    )

    print(
        "False negatives:",
        len(false_negatives)
    )

    # ------------------------------------------------
    # 3. Most common predicted intents in FP/FN
    # ------------------------------------------------

    print("\nFalse-positive predicted intents:")

    print(
        false_positives[
            "predicted_intent"
        ]
        .value_counts()
        .head(10)
    )

    print("\nFalse-negative predicted intents:")

    print(
        false_negatives[
            "predicted_intent"
        ]
        .value_counts()
        .head(10)
    )

    # ------------------------------------------------
    # 4. Lowest confidence intent failures
    # ------------------------------------------------

    print("\n" + "=" * 70)
    print("LOW-CONFIDENCE INTENT FAILURES")
    print("=" * 70)

    print(
        intent_failures[
            [
                "example_id",
                "clean_customer_message",
                "intent",
                "predicted_intent",
                "intent_confidence"
            ]
        ]
        .sort_values("intent_confidence")
        .head(15)
        .to_string(index=False)
    )

    # ------------------------------------------------
    # 5. High-confidence intent failures
    # ------------------------------------------------

    print("\n" + "=" * 70)
    print("HIGH-CONFIDENCE INTENT FAILURES")
    print("=" * 70)

    print(
        intent_failures[
            [
                "example_id",
                "clean_customer_message",
                "intent",
                "predicted_intent",
                "intent_confidence"
            ]
        ]
        .sort_values(
            "intent_confidence",
            ascending=False
        )
        .head(15)
        .to_string(index=False)
    )

    # ------------------------------------------------
    # 6. False positives with strong retrieval
    # ------------------------------------------------

    print("\n" + "=" * 70)
    print("FALSE POSITIVES WITH STRONG RETRIEVAL")
    print("=" * 70)

    print(
        false_positives[
            [
                "example_id",
                "clean_customer_message",
                "expected_escalation",
                "predicted_escalation",
                "intent_confidence",
                "top_retrieval_similarity"
            ]
        ]
        .sort_values(
            "top_retrieval_similarity",
            ascending=False
        )
        .head(15)
        .to_string(index=False)
    )

    # ------------------------------------------------
    # 7. False negatives
    # ------------------------------------------------

    print("\n" + "=" * 70)
    print("FALSE NEGATIVES")
    print("=" * 70)

    print(
        false_negatives[
            [
                "example_id",
                "clean_customer_message",
                "intent",
                "intent_confidence",
                "top_retrieval_similarity",
                "escalation_reason"
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    # ------------------------------------------------
    # 8. Save summary tables
    # ------------------------------------------------

    confusion_counts.reset_index(
        name="count"
    ).to_csv(
        "results/intent_confusion_summary.csv",
        index=False
    )

    false_positives.to_csv(
        "results/escalation_false_positives.csv",
        index=False
    )

    false_negatives.to_csv(
        "results/escalation_false_negatives.csv",
        index=False
    )

    print("\n" + "=" * 70)
    print("SAVED FAILURE ANALYSIS FILES")
    print("=" * 70)

    print("results/intent_confusion_summary.csv")
    print("results/escalation_false_positives.csv")
    print("results/escalation_false_negatives.csv")


if __name__ == "__main__":
    analyze_failures()