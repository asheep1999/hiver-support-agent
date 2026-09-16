def decide_escalation(
    intent,
    confidence,
    historical_cases
):

    reasons = []

    # Get top retrieval similarity
    top_similarity = 0.0

    if historical_cases:
        top_similarity = historical_cases[0]["score"]

    # High-risk intent
    if intent == "apple_id_account_issue":
        return {
            "escalate": True,
            "reason": "Account-related issue requires human review."
        }

    # Explicitly unclear intent
    if intent == "other_or_unclear":
        return {
            "escalate": True,
            "reason": "Customer message does not provide a clear issue."
        }

    # Strong historical evidence can compensate for
    # moderate classifier confidence.
    if (
        confidence >= 0.45
        and top_similarity >= 0.70
    ):
        return {
            "escalate": False,
            "reason": (
                "Intent confidence is acceptable and "
                "strong historical evidence was found."
            )
        }

    # Strong classifier confidence
    if confidence >= 0.70:
        return {
            "escalate": False,
            "reason": (
                "Intent confidence is high and "
                "historical evidence is available."
            )
        }

    # Weak evidence
    if top_similarity < 0.60:
        reasons.append(
            "Historical evidence is weak."
        )

    # Low confidence
    if confidence < 0.45:
        reasons.append(
            "Intent confidence is low."
        )

    return {
        "escalate": True,
        "reason": " ".join(reasons)
    }


if __name__ == "__main__":

    test_cases = [
        {
            "intent": "battery_charging_issue",
            "confidence": 0.7506,
            "historical_cases": [
                {"score": 0.8085},
                {"score": 0.7904},
                {"score": 0.7827}
            ]
        },
        {
            "intent": "apple_id_account_issue",
            "confidence": 0.82,
            "historical_cases": [
                {"score": 0.79}
            ]
        },
        {
            "intent": "other_or_unclear",
            "confidence": 0.35,
            "historical_cases": [
                {"score": 0.42}
            ]
        }
    ]

    for i, case in enumerate(test_cases, 1):

        result = decide_escalation(
            case["intent"],
            case["confidence"],
            case["historical_cases"]
        )

        print(f"\n--- Test Case {i} ---")
        print("Escalate:", result["escalate"])
        print("Reason:", result["reason"])