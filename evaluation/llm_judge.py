import os
import json
import time

import pandas as pd
from dotenv import load_dotenv
from google import genai


# ============================================================
# Configuration
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY was not found. "
        "Check your .env file."
    )

client = genai.Client(api_key=API_KEY)

MODEL_NAME = "gemini-3.6-flash"

INPUT_FILE = "results/reply_quality_cases.csv"
OUTPUT_FILE = "results/reply_quality_cases.csv"


# ============================================================
# LLM Judge
# ============================================================

def judge_reply(
    customer_message,
    historical_evidence,
    generated_reply,
):
    """
    Independently evaluate a generated support reply.

    The judge sees:
    1. The original customer message
    2. The historical evidence given to the generator
    3. The generated reply

    It returns scores from 1-5 for:
    groundedness, relevance, helpfulness, tone, and overall.
    """

    prompt = f"""
You are an evaluator judging the quality of an AI-generated
customer support response for AppleSupport.

You must evaluate the generated reply using ONLY the information
provided in this evaluation prompt.

============================================================
CUSTOMER MESSAGE
============================================================

{customer_message}

============================================================
HISTORICAL APPLESUPPORT EVIDENCE
============================================================

{historical_evidence}

============================================================
GENERATED REPLY
============================================================

{generated_reply}

============================================================
SCORING CRITERIA
============================================================

1. GROUNDEDNESS

Question:
Is the generated reply supported by the historical AppleSupport
evidence?

5 = Fully supported by the evidence. No meaningful unsupported claims.
4 = Mostly supported, with only a very minor generic addition.
3 = Partially supported; some content is generic or weakly supported.
2 = Contains substantial unsupported content.
1 = Clearly invents advice, claims, policies, procedures, or facts.

2. RELEVANCE

Question:
Does the reply address the customer's actual problem?

5 = Directly addresses the customer's specific problem.
4 = Relevant but somewhat generic.
3 = Partially addresses the problem.
2 = Barely addresses the problem.
1 = Unrelated to the customer's problem.

3. HELPFULNESS

Question:
Does the reply meaningfully move the support conversation forward?

5 = Very useful and gives an appropriate next step.
4 = Useful and reasonably actionable.
3 = Somewhat useful but limited.
2 = Provides little practical help.
1 = Provides no meaningful help.

4. TONE

Question:
Is the reply appropriate for a professional AppleSupport-style
customer interaction?

5 = Professional, concise, calm, empathetic, and natural.
4 = Good professional tone with minor weaknesses.
3 = Acceptable but somewhat generic or robotic.
2 = Awkward or noticeably inappropriate.
1 = Unprofessional or inappropriate.

5. OVERALL

Question:
Considering all four dimensions, how good is the reply overall?

5 = Excellent.
4 = Good.
3 = Acceptable.
2 = Weak.
1 = Unacceptable.

============================================================
IMPORTANT EVALUATION RULES
============================================================

- Judge the generated reply independently.
- Do NOT try to agree with any human score.
- Do NOT assume the generated reply is correct.
- Groundedness must be judged against the HISTORICAL EVIDENCE.
- Do not penalize a reply merely because it does not contain every
  detail from the historical evidence.
- Do penalize unsupported troubleshooting instructions or claims.
- Do not reward invented Apple policies, procedures, prices, timelines,
  refunds, replacements, URLs, or guarantees.
- The customer's actual problem matters when judging relevance.
- Keep the scoring consistent across all cases.

Return ONLY valid JSON.

Use exactly this structure:

{{
    "groundedness": 1,
    "relevance": 1,
    "helpfulness": 1,
    "tone": 1,
    "overall": 1
}}

All values must be integers from 1 to 5.
"""

    interaction = client.interactions.create(
        model=MODEL_NAME,
        input=prompt,
    )

    response_text = interaction.output_text.strip()

    # Remove markdown JSON fences if Gemini adds them.
    if response_text.startswith("```"):
        response_text = response_text.replace("```json", "")
        response_text = response_text.replace("```", "")
        response_text = response_text.strip()

    result = json.loads(response_text)

    required_keys = [
        "groundedness",
        "relevance",
        "helpfulness",
        "tone",
        "overall",
    ]

    for key in required_keys:
        if key not in result:
            raise ValueError(
                f"Judge response is missing required field: {key}"
            )

        if not isinstance(result[key], int):
            raise ValueError(
                f"Judge score for {key} is not an integer: "
                f"{result[key]}"
            )

        if result[key] < 1 or result[key] > 5:
            raise ValueError(
                f"Judge score for {key} is outside 1-5: "
                f"{result[key]}"
            )

    return result


# ============================================================
# Main evaluation
# ============================================================

def main():

    print("=" * 70)
    print("LLM JUDGE - REPLY QUALITY EVALUATION")
    print("=" * 70)

    print(f"\nReading: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print(f"Total cases: {len(df)}")

    # Make sure judge columns exist.
    judge_columns = [
        "judge_groundedness",
        "judge_relevance",
        "judge_helpfulness",
        "judge_tone",
        "judge_overall",
    ]

    for column in judge_columns:
        if column not in df.columns:
            df[column] = pd.NA

    # Identify routed-to-human cases.
    routed_mask = (
        df["generated_reply"]
        .fillna("")
        .astype(str)
        .str.contains(
            r"\[ROUTED TO HUMAN\]",
            case=False,
            regex=True,
        )
    )

    routed_count = routed_mask.sum()
    generated_count = len(df) - routed_count

    print(f"Generated replies to judge: {generated_count}")
    print(f"Routed-to-human cases skipped: {routed_count}")

    print("\nStarting LLM judging...")
    print("This will make one Gemini call per generated reply.\n")

    judged_count = 0

    for index, row in df.iterrows():

        # Skip routed-to-human cases.
        if routed_mask.loc[index]:
            print(
                f"Case {row['example_id']}: "
                "SKIPPED - routed to human"
            )
            continue

        # If all judge scores already exist, do not call Gemini again.
        existing_scores = [
            row[column]
            for column in judge_columns
        ]

        if all(pd.notna(score) for score in existing_scores):
            print(
                f"Case {row['example_id']}: "
                "ALREADY JUDGED - skipped"
            )
            continue

        customer_message = str(
            row.get("clean_customer_message", "")
        )

        historical_evidence = str(
            row.get("historical_evidence", "")
        )

        generated_reply = str(
            row.get("generated_reply", "")
        )

        print(
            f"Case {row['example_id']}: "
            "judging..."
        )

        try:

            result = judge_reply(
                customer_message=customer_message,
                historical_evidence=historical_evidence,
                generated_reply=generated_reply,
            )

            df.loc[index, "judge_groundedness"] = result[
                "groundedness"
            ]

            df.loc[index, "judge_relevance"] = result[
                "relevance"
            ]

            df.loc[index, "judge_helpfulness"] = result[
                "helpfulness"
            ]

            df.loc[index, "judge_tone"] = result[
                "tone"
            ]

            df.loc[index, "judge_overall"] = result[
                "overall"
            ]

            judged_count += 1

            print(
                f"  Groundedness: {result['groundedness']}"
            )
            print(
                f"  Relevance:    {result['relevance']}"
            )
            print(
                f"  Helpfulness:  {result['helpfulness']}"
            )
            print(
                f"  Tone:         {result['tone']}"
            )
            print(
                f"  Overall:      {result['overall']}"
            )

            # Small delay between requests.
            time.sleep(1)

        except Exception as e:

            print(
                f"  ERROR while judging case "
                f"{row['example_id']}: {e}"
            )

            # Save progress before stopping.
            df.to_csv(
                OUTPUT_FILE,
                index=False,
            )

            raise

    # Save final results.
    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n" + "=" * 70)
    print("LLM JUDGE COMPLETE")
    print("=" * 70)

    print(f"Cases judged: {judged_count}")
    print(f"Results saved to: {OUTPUT_FILE}")

    print("\nJudge scores:")
    print(
        df[
            [
                "example_id",
                "judge_groundedness",
                "judge_relevance",
                "judge_helpfulness",
                "judge_tone",
                "judge_overall",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()