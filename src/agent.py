import pandas as pd

from src.classifier import build_model
from src.retrieval import AppleSupportRetriever
from src.escalation import decide_escalation
from src.generator import generate_reply


DEV_PATH = "data/processed/apple_support_dev.csv"


def train_classifier():
    df = pd.read_csv(DEV_PATH)

    X = df["clean_customer_message"].astype(str)
    y = df["intent"].astype(str)

    model = build_model()
    model.fit(X, y)

    return model


class AppleSupportAgent:

    def __init__(self):
        print("Loading intent classifier...")
        self.classifier = train_classifier()

        print("Loading semantic retriever...")
        self.retriever = AppleSupportRetriever()

    def analyze(self, customer_message, top_k=3):

        # 1. Intent classification
        intent = self.classifier.predict(
            [customer_message]
        )[0]

        probabilities = self.classifier.predict_proba(
            [customer_message]
        )[0]

        confidence = float(probabilities.max())

        # 2. Historical retrieval
        historical_cases = self.retriever.search(
            customer_message,
            top_k=top_k
        )

        # 3. Escalation decision
        escalation = decide_escalation(
            intent=intent,
            confidence=confidence,
            historical_cases=historical_cases
        )

        result = {
            "customer_message": customer_message,
            "intent": intent,
            "confidence": confidence,
            "historical_cases": historical_cases,
            "escalate": escalation["escalate"],
            "escalation_reason": escalation["reason"],
            "reply": None
        }

        # 4. Generate a reply ONLY if safe to auto-handle
        if not escalation["escalate"]:

            result["reply"] = generate_reply(
                customer_message=customer_message,
                intent=intent,
                historical_cases=historical_cases
            )

        return result


if __name__ == "__main__":

    agent = AppleSupportAgent()

    test_messages = [
        "My iPhone battery is draining very quickly after the latest iOS update.",
        "My keyboard changes the letter I into a strange symbol.",
        "I cannot recover my Apple ID account.",
        "My iPhone keeps freezing and becoming very slow.",
        "Help me."
    ]

    for message in test_messages:

        result = agent.analyze(message)

        print("\n" + "=" * 70)
        print("CUSTOMER")
        print("=" * 70)
        print(result["customer_message"])

        print("\nINTENT:")
        print(result["intent"])

        print("\nCONFIDENCE:")
        print(round(result["confidence"], 4))

        print("\nESCALATE:")
        print(result["escalate"])

        print("\nESCALATION REASON:")
        print(result["escalation_reason"])

        if result["reply"]:

            print("\nGENERATED REPLY:")
            print(result["reply"])

        else:

            print("\nGENERATED REPLY:")
            print("None — routed to human support.")