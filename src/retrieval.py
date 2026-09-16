import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
import faiss


DATA_PATH = "data/processed/apple_support_pairs.csv"


class AppleSupportRetriever:

    def __init__(self):
        print("Loading historical AppleSupport conversations...")

        self.df = pd.read_csv(DATA_PATH)

        self.df["customer_message"] = (
            self.df["customer_message"]
            .fillna("")
            .astype(str)
        )

        self.df["support_response"] = (
            self.df["support_response"]
            .fillna("")
            .astype(str)
        )

        print("Historical examples:", len(self.df))

        # Small and fast sentence embedding model
        self.encoder = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        print("Creating embeddings...")

        embeddings = self.encoder.encode(
            self.df["customer_message"].tolist(),
            show_progress_bar=True,
            normalize_embeddings=True
        )

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        self.embeddings = embeddings

        # Because vectors are normalized,
        # inner product ~= cosine similarity
        self.index = faiss.IndexFlatIP(
            embeddings.shape[1]
        )

        self.index.add(embeddings)

        print("FAISS index created.")
        print("Indexed examples:", self.index.ntotal)

    def search(self, query, top_k=3):

        query_embedding = self.encoder.encode(
            [query],
            normalize_embeddings=True
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):

            row = self.df.iloc[int(idx)]

            results.append({
                "score": float(score),
                "customer_message": row["customer_message"],
                "support_response": row["support_response"]
            })

        return results


if __name__ == "__main__":

    retriever = AppleSupportRetriever()

    test_queries = [
        "My keyboard changes the letter I into a strange symbol.",
        "My iPhone keeps freezing and becoming very slow.",
        "My Apple Watch is not counting my activity correctly.",
        "I cannot recover my Apple ID account.",
        "My Wi-Fi keeps disconnecting on my iPhone."
    ]

    for query in test_queries:

        print("\n" + "=" * 70)
        print("QUERY")
        print("=" * 70)
        print(query)

        results = retriever.search(
            query,
            top_k=3
        )

        print("\nTOP 3 MATCHES:")

        for i, result in enumerate(results, 1):

            print(f"\n--- Result {i} ---")
            print("Similarity:", round(result["score"], 4))
            print("Customer:", result["customer_message"])
            print("AppleSupport:", result["support_response"])