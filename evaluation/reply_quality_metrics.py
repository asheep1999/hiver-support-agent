import pandas as pd
import numpy as np


INPUT_FILE = "results/reply_quality_cases.csv"


def main():

    print("=" * 70)
    print("REPLY QUALITY - FINAL EVALUATION")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)

    # --------------------------------------------------------
    # Columns
    # --------------------------------------------------------

    human_columns = [
        "human_groundedness",
        "human_relevance",
        "human_helpfulness",
        "human_tone",
        "human_overall",
    ]

    judge_columns = [
        "judge_groundedness",
        "judge_relevance",
        "judge_helpfulness",
        "judge_tone",
        "judge_overall",
    ]

    # Convert scores to numeric.
    # "NA" / blank values become NaN.
    for column in human_columns + judge_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Human evaluation
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("HUMAN EVALUATION")
    print("-" * 70)

    human_means = df[human_columns].mean()

    for column, value in human_means.items():
        print(
            f"{column}: {value:.2f}"
        )

    print(
        f"\nHuman overall mean: "
        f"{human_means['human_overall']:.2f} / 5"
    )

    # --------------------------------------------------------
    # LLM judge evaluation
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("LLM JUDGE EVALUATION")
    print("-" * 70)

    judge_means = df[judge_columns].mean()

    for column, value in judge_means.items():
        print(
            f"{column}: {value:.2f}"
        )

    print(
        f"\nLLM judge overall mean: "
        f"{judge_means['judge_overall']:.2f} / 5"
    )

    # --------------------------------------------------------
    # Agreement
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("HUMAN vs LLM JUDGE AGREEMENT")
    print("-" * 70)

    pairs = [
        (
            "Groundedness",
            "human_groundedness",
            "judge_groundedness",
        ),
        (
            "Relevance",
            "human_relevance",
            "judge_relevance",
        ),
        (
            "Helpfulness",
            "human_helpfulness",
            "judge_helpfulness",
        ),
        (
            "Tone",
            "human_tone",
            "judge_tone",
        ),
        (
            "Overall",
            "human_overall",
            "judge_overall",
        ),
    ]

    agreement_results = []

    for name, human_col, judge_col in pairs:

        comparison = df[
            [human_col, judge_col]
        ].dropna()

        if len(comparison) == 0:
            print(
                f"{name}: no comparable cases"
            )
            continue

        exact_matches = (
            comparison[human_col]
            == comparison[judge_col]
        ).sum()

        total = len(comparison)

        exact_agreement = (
            exact_matches / total
        )

        mean_absolute_difference = (
            comparison[human_col]
            - comparison[judge_col]
        ).abs().mean()

        # Agreement within one point.
        within_one = (
            (
                comparison[human_col]
                - comparison[judge_col]
            ).abs()
            <= 1
        ).mean()

        print(f"\n{name}")
        print(f"  Comparable cases: {total}")
        print(
            f"  Exact agreement: "
            f"{exact_matches}/{total} "
            f"({exact_agreement * 100:.1f}%)"
        )
        print(
            f"  Agreement within ±1: "
            f"{within_one * 100:.1f}%"
        )
        print(
            f"  Mean absolute difference: "
            f"{mean_absolute_difference:.2f}"
        )

        agreement_results.append(
            {
                "dimension": name,
                "comparable_cases": total,
                "exact_agreement_percent":
                    exact_agreement * 100,
                "within_one_percent":
                    within_one * 100,
                "mean_absolute_difference":
                    mean_absolute_difference,
            }
        )

    # --------------------------------------------------------
    # Overall comparison
    # --------------------------------------------------------

    comparison = df[
        [
            "human_overall",
            "judge_overall",
        ]
    ].dropna()

    if len(comparison) > 0:

        exact_overall = (
            comparison["human_overall"]
            == comparison["judge_overall"]
        ).mean()

        within_one_overall = (
            (
                comparison["human_overall"]
                - comparison["judge_overall"]
            ).abs()
            <= 1
        ).mean()

        mean_difference = (
            comparison["human_overall"]
            - comparison["judge_overall"]
        ).abs().mean()

        print("\n" + "-" * 70)
        print("OVERALL AGREEMENT SUMMARY")
        print("-" * 70)

        print(
            f"Comparable cases: {len(comparison)}"
        )

        print(
            f"Exact agreement: "
            f"{exact_overall * 100:.1f}%"
        )

        print(
            f"Agreement within ±1 point: "
            f"{within_one_overall * 100:.1f}%"
        )

        print(
            f"Mean absolute difference: "
            f"{mean_difference:.2f}"
        )

    # --------------------------------------------------------
    # Per-case comparison table
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("PER-CASE OVERALL COMPARISON")
    print("-" * 70)

    print(
        df[
            [
                "example_id",
                "human_overall",
                "judge_overall",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        f"Human overall: "
        f"{human_means['human_overall']:.2f} / 5"
    )

    print(
        f"LLM judge overall: "
        f"{judge_means['judge_overall']:.2f} / 5"
    )

    if len(comparison) > 0:
        print(
            f"Overall exact agreement: "
            f"{exact_overall * 100:.1f}%"
        )

        print(
            f"Overall agreement within ±1: "
            f"{within_one_overall * 100:.1f}%"
        )

        print(
            f"Overall mean absolute difference: "
            f"{mean_difference:.2f}"
        )

    print("\nNo files were modified.")


if __name__ == "__main__":
    main()