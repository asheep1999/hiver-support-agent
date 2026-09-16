# Decision Log

## 1. Selected AppleSupport as the target brand
Reason: Sufficient historical conversations were available in the sampled dataset.

## 2. Used conversation pairs rather than isolated tweets
Reason: Customer messages need corresponding historical support replies for grounded generation.

## 3. Used 100k rows for initial development
Reason: Full 3M-row processing was unnecessary for the assignment and would increase iteration time.

## 4. Created a 10-intent taxonomy
Reason: The assignment asks for a small set of practical customer-support intents.

## 5. Kept other_or_unclear as a class
Reason: Some historical customer messages do not provide enough information for reliable routing.

## 6. Protected 200 human-labelled examples as the golden set
Reason: Evaluation must not depend on weak labels.

## 7. Used weak labels only for development training
Reason: Hand-labeling the entire dataset would be inefficient.

## 8. Used TF-IDF + Logistic Regression as a baseline
Reason: Simple, fast and interpretable baseline.

## 9. Added semantic retrieval
Reason: Historical support conversations provide evidence for grounded responses.

## 10. Used retrieval together with classification
Reason: Intent alone cannot provide the content needed for a support response.

## 11. Added an escalation mechanism
Reason: Low-confidence or risky requests should be routed to humans.

## 12. Did not use fine-tuning
Reason: The assignment prioritizes a practical, reproducible system rather than expensive model training.

## 13. Limited reply-quality evaluation to a representative subset
Reason: Generative API quotas made full-scale generation impractical.