import pandas as pd
import os

RESULTS_PATH = "results/agent_predictions.csv"
OUTPUT_PATH = "results/reply_quality_cases.csv"


def select_cases():

    df = pd.read_csv(RESULTS_PATH)

    # -----------------------------------------
    # Easy / high-confidence cases
    # -----------------------------------------

    easy = (
        df[
            (df["predicted_escalation"] == "no") &
            (df["intent_confidence"] >= 0.80) &
            (df["top_retrieval_similarity"] >= 0.75) &
            (df["intent"] == df["predicted_intent"])
        ]
        .sort_values(
            ["intent_confidence", "top_retrieval_similarity"],
            ascending=False
        )
        .head(5)
    )

    # -----------------------------------------
    # Difficult / ambiguous cases
    # -----------------------------------------

    difficult = (
        df[
            (df["predicted_escalation"] == "no") &
            (
                (df["intent_confidence"] < 0.60) |
                (df["top_retrieval_similarity"] < 0.70)
            )
        ]
        .sort_values("intent_confidence")
        .head(5)
    )

    # -----------------------------------------
    # Actual intent failures
    # -----------------------------------------

    failures = (
        df[
            df["intent"] != df["predicted_intent"]
        ]
        .sort_values("intent_confidence")
        .head(5)
    )

    # Combine
    selected = pd.concat(
        [easy, difficult, failures],
        ignore_index=True
    )

    # Remove accidental duplicates
    selected = selected.drop_duplicates(
        subset=["example_id"]
    )

    # Keep useful columns
    selected = selected[
        [
            "example_id",
            "clean_customer_message",
            "intent",
            "predicted_intent",
            "intent_confidence",
            "top_retrieval_similarity",
            "expected_escalation",
            "predicted_escalation"
        ]
    ]

    # Add columns for later reply evaluation
    selected["historical_evidence"] = ""
    selected["generated_reply"] = ""

    selected["human_groundedness"] = ""
    selected["human_relevance"] = ""
    selected["human_helpfulness"] = ""
    selected["human_tone"] = ""
    selected["human_overall"] = ""

    selected["judge_groundedness"] = ""
    selected["judge_relevance"] = ""
    selected["judge_helpfulness"] = ""
    selected["judge_tone"] = ""
    selected["judge_overall"] = ""

    os.makedirs("results", exist_ok=True)

    selected.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("=" * 70)
    print("REPLY QUALITY CASES")
    print("=" * 70)

    print("Selected cases:", len(selected))

    print("\nSelected example IDs:")
    print(selected["example_id"].tolist())

    print("\nSaved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    select_cases()