import os
import pandas as pd


RESULTS_DIR = "results"

INTENT_PATH = os.path.join(
    RESULTS_DIR,
    "intent_confusion_summary.csv"
)

FP_PATH = os.path.join(
    RESULTS_DIR,
    "escalation_false_positives.csv"
)

FN_PATH = os.path.join(
    RESULTS_DIR,
    "escalation_false_negatives.csv"
)

OUTPUT_PATH = os.path.join(
    RESULTS_DIR,
    "failure_analysis.csv"
)


def main():

    required_files = [
        INTENT_PATH,
        FP_PATH,
        FN_PATH
    ]

    for path in required_files:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    intent_df = pd.read_csv(INTENT_PATH)
    fp_df = pd.read_csv(FP_PATH)
    fn_df = pd.read_csv(FN_PATH)

    rows = []

    # Intent classification failures
    for _, row in intent_df.iterrows():
        rows.append({
            "failure_type": "intent_confusion",
            "details": (
                f"Actual intent '{row.iloc[0]}' "
                f"was predicted as '{row.iloc[1]}' "
                f"{row.iloc[2]} time(s)."
            )
        })

    # Escalation false positives
    for _, row in fp_df.iterrows():
        rows.append({
            "failure_type": "escalation_false_positive",
            "details": (
                f"Example {row.get('example_id', 'unknown')} "
                f"was escalated although the gold label "
                f"expected no escalation."
            )
        })

    # Escalation false negatives
    for _, row in fn_df.iterrows():
        rows.append({
            "failure_type": "escalation_false_negative",
            "details": (
                f"Example {row.get('example_id', 'unknown')} "
                f"was not escalated although the gold label "
                f"expected escalation."
            )
        })

    output_df = pd.DataFrame(rows)

    output_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("=" * 60)
    print("COMBINED FAILURE ANALYSIS")
    print("=" * 60)
    print(f"Rows written: {len(output_df)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()