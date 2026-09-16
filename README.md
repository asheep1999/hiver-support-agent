# Hiver SDE Intern Take-Home
## AI Customer Support Agent for AppleSupport

## 1. Problem Framing

This project implements an AI customer-support agent using historical customer-support conversations from the Customer Support on Twitter dataset.

The system focuses on the AppleSupport brand and performs three tasks:

1. Classifies an incoming customer message into a small set of support intents.
2. Retrieves historically similar AppleSupport conversations and uses them as evidence for generating a support response.
3. Decides whether the request should be automatically handled or escalated to a human.

The goal is to build a small, reproducible support-agent pipeline rather than a general-purpose chatbot.

---

## 2. Dataset

The project uses the Customer Support on Twitter dataset.

Relevant fields include:

- tweet_id
- author_id
- inbound
- created_at
- text
- response_tweet_id
- in_response_to_tweet_id

AppleSupport was selected as the target support brand.

A bounded 100,000-row sample was used during development rather than processing the complete dataset.

From the sampled data, 3,091 usable historical customer/support response pairs were reconstructed.

After cleaning and removing unusable customer messages, 3,069 examples remained for development and evaluation preparation.

---

## 3. Intent Taxonomy

The system uses ten intents:

| Intent | Description |
|---|---|
| ios_update_issue | Problems directly related to installing or updating iOS |
| app_or_feature_issue | Problems involving apps or device features |
| battery_charging_issue | Battery drain, battery performance or charging problems |
| keyboard_autocorrect_issue | Keyboard, typing or autocorrect problems |
| device_slow_freezing_crashing | Slow performance, freezing or crashing |
| wifi_mobile_data_issue | Wi-Fi, cellular-data or connectivity problems |
| imessage_facetime_issue | iMessage or FaceTime problems |
| apple_id_account_issue | Apple ID, account access or account-related problems |
| apple_watch_issue | Apple Watch-related problems |
| other_or_unclear | Messages without enough information for a specific intent |

The taxonomy was intentionally kept small so that classification and evaluation remain interpretable.

---

## 4. Data Splitting and Golden Set

A protected 200-example golden set was manually labelled.

The golden set was not used for classifier training.

The remaining 2,869 examples were used as development training data and received weak labels using keyword-based rules.

Therefore:

- development data = weakly labelled
- golden set = human labelled
- final classification evaluation = protected human-labelled golden set

This separation prevents the classifier from being evaluated against the same weak labels used during development.

---

## 5. System Architecture

```text
Customer Message
       |
       v
Text Cleaning
       |
       v
Intent Classifier
       |
       +----------------------+
       |                      |
       v                      v
Intent Confidence     Historical Retrieval
                              |
                              v
                       Similar Cases
                              |
                              v
                       Escalation Policy
                          /          \
                         /            \
                  Escalate          Auto-handle
                     |                 |
                     v                 v
                Human Review      Grounded Reply


---

## 6. Classification Model

The first baseline is a majority-class classifier.

The majority class is:

`other_or_unclear`

The second baseline is:

`TF-IDF + Logistic Regression`

The classifier uses word unigrams and bigrams with class-balanced Logistic Regression.

All headline classification metrics are measured against the protected 200-example golden set.

---

## 7. Retrieval

Historical AppleSupport customer/support pairs are embedded using:

`all-MiniLM-L6-v2`

The embeddings are normalized and indexed using FAISS `IndexFlatIP`.

For each incoming customer message, the retriever returns semantically similar historical conversations.

The retrieved examples are used as evidence for response generation.

---

## 8. Response Generation

For requests that are not escalated, the generator receives:

- The customer message
- The predicted intent
- Retrieved historical customer messages
- Corresponding historical support responses

The generation prompt instructs the model to:

- Stay grounded in historical evidence
- Directly address the customer's issue
- Avoid inventing policies, prices, refunds or eligibility rules
- Avoid unsupported fixes
- Ask for more information when evidence is insufficient
- Maintain a concise and professional support tone

The generator is therefore used as a grounded support-response writer.

---

## 9. Escalation

The agent can route a customer request to a human rather than automatically generating a reply.

The current policy escalates:

- Account-related requests
- Unclear requests
- Cases with insufficient classifier confidence
- Cases with insufficient historical retrieval evidence

The intention is to avoid automatically producing unsupported responses for uncertain or sensitive cases.

---

# 10. Evaluation

## 10.1 Majority Baseline

| Metric | Result |
|---|---:|
| Accuracy | 0.325 |
| Macro-F1 | 0.05 |

## 10.2 TF-IDF + Logistic Regression

| Metric | Result |
|---|---:|
| Accuracy | 0.61 |
| Macro-F1 | 0.58 |
| Weighted-F1 | 0.62 |

The complete agent currently uses this classifier, so the intent-classification results for the full agent are also:

| Metric | Result |
|---|---:|
| Accuracy | 0.61 |
| Macro-F1 | 0.58 |

The retrieval and response-generation stages provide functionality beyond the classifier and are evaluated separately.

---

# 11. Escalation Results

The current escalation policy achieved the following results on the 200-example golden set:

| Metric | Result |
|---|---:|
| Precision | 0.50 |
| Recall | 0.8941 |
| F1 | 0.6414 |
| Accuracy | 0.57 |

The system produced:

- 76 false positives
- 9 false negatives

This means the current policy is conservative: it catches most examples marked for escalation but also routes a substantial number of cases to human review unnecessarily.

---

# 12. Failure Analysis

## Failure Mode 1: `other_or_unclear` absorbs clear issues

The classifier sometimes predicts `other_or_unclear` even when the message contains enough information for a more specific category.

The most common confusion includes:

`app_or_feature_issue → other_or_unclear`

and

`other_or_unclear → app_or_feature_issue`

This suggests that the miscellaneous category is broad and heterogeneous.

---

## Failure Mode 2: iOS update context is confused with the actual problem

Messages mentioning an iOS update are sometimes classified as `ios_update_issue` even when the actual problem involves another symptom such as battery, freezing or an application.

This indicates a strong dependence on lexical signals such as "iOS", "update" and software-version numbers.

---

## Failure Mode 3: Rare intents have limited evaluation coverage

Some intents have very few examples in the protected golden set.

Examples:

- `imessage_facetime_issue`: 1
- `apple_id_account_issue`: 3
- `apple_watch_issue`: 4
- `wifi_mobile_data_issue`: 4

Therefore, class-specific conclusions for these categories have high uncertainty.

---

## Failure Mode 4: Escalation produces many false positives

The escalation policy produced:

- 76 false positives
- 9 false negatives

Some cases had strong historical retrieval similarity but were still escalated because classifier confidence was low.

This indicates that the current threshold-based policy does not fully reconcile classifier confidence with retrieval evidence.

---

## Failure Mode 5: Multiple symptoms are difficult to represent

Some customer messages contain several symptoms or both context and symptom information.

The current system assigns one primary intent, so these cases may be represented imperfectly.

A multi-label or symptom-plus-context representation could better capture such messages.

---

# 13. What Is Misleading About My Headline Number?

The 61% accuracy is useful as a summary metric, but it should not be interpreted as uniform performance across all intents.

The protected golden set contains 200 human-labelled examples and is imbalanced.

`other_or_unclear` represents a large portion of the golden set, while several other intents contain only a few examples.

Therefore, accuracy can hide uncertainty or weaker performance for rare categories.

Macro-F1 is reported alongside accuracy to provide a more balanced view across classes.

The confusion analysis also shows substantial confusion between:

- `other_or_unclear`
- `app_or_feature_issue`
- `ios_update_issue`

For this reason, the headline number is reported together with class-level results and concrete failure examples.

---

# 14. Reply Quality Evaluation

Reply quality is evaluated separately from classification.

A representative set of 15 examples is used, containing:

- Straightforward high-confidence cases
- Difficult cases
- Classification-failure cases

Each generated response is evaluated on:

| Criterion | Definition |
|---|---|
| Groundedness | Is the response supported by retrieved historical evidence? |
| Relevance | Does it directly address the customer's issue? |
| Helpfulness | Does it provide an appropriate next step? |
| Tone | Is it professional and suitable for customer support? |
| Overall | Would the response be acceptable as a support reply? |

Each criterion uses a 1–5 scale.

Human evaluation is performed first.

A smaller subset is also evaluated by an LLM judge using the same rubric.

Judge-human agreement is reported separately and is not treated as ground truth.

Reply-quality results will be added after the generative evaluation stage.

---

# 15. What I Would Improve With One More Week

### Improve human-labelled training data

Increase the number of manually labelled examples, especially for rare intents.

### Improve intent representation

Separate context such as "after an iOS update" from the actual symptom such as battery drain, freezing or connectivity.

### Improve escalation

Use calibrated confidence and a learned decision boundary instead of relying only on fixed thresholds.

### Improve retrieval

Test different embedding models and evaluate retrieval quality systematically.

### Improve response evaluation

Use more human raters and a larger reply-quality evaluation set.

---

# 16. Project Structure

```text
hiver-support-agent/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── golden/
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── conversation_builder.py
│   ├── intent_discovery.py
│   ├── classifier.py
│   ├── retrieval.py
│   ├── generator.py
│   ├── escalation.py
│   └── agent.py
│
├── evaluation/
│   ├── __init__.py
│   ├── evaluate_agent.py
│   ├── evaluate_intent.py
│   ├── evaluate_escalation.py
│   ├── failure_analysis.py
│   ├── combine_failure_analysis.py
│   ├── create_confusion_matrix.py
│   ├── select_reply_cases.py
│   └── llm_judge.py
│
├── results/
│   ├── agent_predictions.csv
│   ├── confusion_matrix.png
│   ├── escalation_false_negatives.csv
│   ├── escalation_false_positives.csv
│   ├── failure_analysis.csv
│   ├── intent_confusion_summary.csv
│   ├── metrics.json
│   └── reply_quality_cases.csv
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── README.md
├── DECISION_LOG.md
├── requirements.txt
├── .gitignore
└── .env

# 17. Reproduction

## Install dependencies

Create a virtual environment:

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the dataset at:

```bash
data/raw/twcs.csv
```

The raw dataset is excluded from version control.

Create a .env file:

```bash
GEMINI_API_KEY=YOUR_API_KEY
```

The API key must not be committed.

Run the evaluation:

```bash
python -m evaluation.evaluate_agent
```

Generate the confusion matrix:

```bash
python evaluation/create_confusion_matrix.py
```

Generate failure analysis:

```bash
python evaluation/failure_analysis.py
```


# 18. Current Status

The classification, retrieval, escalation and generative response pipeline is implemented and evaluated.

The protected golden set contains 200 human-labelled examples.

## Final Reply-Quality Evaluation

A representative subset of 15 cases was used for generative evaluation because generative API calls are quota-limited.

- 10 cases produced generated replies and were evaluated by a human evaluator and an independent LLM judge.
- 5 cases were routed to a human and were excluded from generative reply-quality scoring.

### Human Evaluation

| Dimension | Mean Score |
|---|---:|
| Groundedness | 3.70 / 5 |
| Relevance | 3.10 / 5 |
| Helpfulness | 2.90 / 5 |
| Tone | 3.90 / 5 |
| Overall | 3.20 / 5 |

### LLM Judge Evaluation

| Dimension | Mean Score |
|---|---:|
| Groundedness | 5.00 / 5 |
| Relevance | 4.80 / 5 |
| Helpfulness | 4.80 / 5 |
| Tone | 5.00 / 5 |
| Overall | 4.80 / 5 |

### Human vs LLM Judge Agreement

The comparison was performed on the 10 cases that received both human and LLM-judge scores.

| Dimension | Exact Agreement | Agreement Within ±1 | Mean Absolute Difference |
|---|---:|---:|---:|
| Groundedness | 0.0% | 80.0% | 1.30 |
| Relevance | 0.0% | 30.0% | 1.70 |
| Helpfulness | 0.0% | 20.0% | 1.90 |
| Tone | 10.0% | 80.0% | 1.10 |
| Overall | 0.0% | 40.0% | 1.60 |

The human evaluator's overall mean was 3.20/5, compared with 4.80/5 from the LLM judge. Overall exact agreement was 0.0%, while 40.0% of cases were within ±1 point.

This indicates that the LLM judge was substantially more optimistic than the human evaluation on this small representative sample. Therefore, human evaluation is treated as the primary signal for reply-quality assessment, while the LLM judge is reported as a secondary automated evaluation.

## Current Limitations

- The generative evaluation uses a small 15-case representative subset because of API quota constraints.
- Only 10 cases received generated replies; 5 cases were routed to a human.
- The human-vs-LLM agreement results are based on only 10 comparable cases.
- Rare intents have limited training examples.
- The current classifier uses a single primary intent for each customer message, which can be challenging for multi-symptom requests.
- Escalation produces some false positives.
- Reply quality remains sensitive to the quality and relevance of retrieved historical evidence.