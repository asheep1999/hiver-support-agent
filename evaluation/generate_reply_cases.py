import os
import pandas as pd

from src.classifier import build_model
from src.retrieval import AppleSupportRetriever
from src.escalation import decide_escalation
from src.generator import generate_reply


DEV_PATH = "data/processed/apple_support_dev.csv"
CASES_PATH = "results/reply_quality_cases.csv"
OUTPUT_PATH = "results/reply_quality_cases.csv"


def train_classifier():
    df = pd.read_csv(DEV_PATH)

    X = df["clean_customer_message"].astype(str)
    y = df["intent"].astype(str)

    model = build_model()
    model.fit(X, y)

    return model


def generate_cases():

    cases = pd.read_csv(CASES_PATH)

    # Force text columns to object dtype so we can safely write strings.
    # Empty/all-NaN columns get read in as float64 by default, which then
    # raises LossySetitemError the moment we try to write a string into them.
    text_columns = [
        "predicted_intent",
        "predicted_escalation",
        "generated_reply",
        "historical_evidence",
    ]
    for col in text_columns:
        if col not in cases.columns:
            cases[col] = ""
        cases[col] = cases[col].astype(object)

    classifier = train_classifier()
    retriever = AppleSupportRetriever()

    for index, row in cases.iterrows():

        # Skip if reply already exists
        if (
            pd.notna(row["generated_reply"])
            and str(row["generated_reply"]).strip() != ""
        ):
            continue

        message = str(row["clean_customer_message"])

        intent = classifier.predict([message])[0]

        probabilities = classifier.predict_proba([message])[0]
        confidence = float(probabilities.max())

        historical_cases = retriever.search(
            message,
            top_k=3
        )

        escalation = decide_escalation(
            intent=intent,
            confidence=confidence,
            historical_cases=historical_cases
        )

        # Save the actual system decision
        cases.loc[index, "predicted_intent"] = intent
        cases.loc[index, "intent_confidence"] = confidence
        cases.loc[index, "predicted_escalation"] = (
            "yes" if escalation["escalate"] else "no"
        )

        # Don't generate replies for escalated cases
        if escalation["escalate"]:
            cases.loc[index, "generated_reply"] = (
                "[ROUTED TO HUMAN]"
            )

            continue

        # Save historical evidence
        evidence = []

        for i, case in enumerate(
            historical_cases,
            start=1
        ):
            evidence.append(
                f"Case {i}\n"
                f"Customer: {case['customer_message']}\n"
                f"AppleSupport: {case['support_response']}"
            )

        cases.loc[index, "historical_evidence"] = (
            "\n\n".join(evidence)
        )

        print("\nGenerating reply for:")
        print(message)

        reply = generate_reply(
            customer_message=message,
            intent=intent,
            historical_cases=historical_cases
        )

        cases.loc[index, "generated_reply"] = reply

        # Save immediately after every successful API call
        cases.to_csv(
            OUTPUT_PATH,
            index=False
        )

    cases.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nSaved:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    generate_cases()