import os
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest


PCA_FILE = "data/difficulty_pca.npy"
INPUT_CSV = "data/issues_with_hybrid_concepts.csv"

OUTPUT_CSV = "data/isolation_forest_difficulty.csv"


def main():
    print("Loading PCA features...")

    X = np.load(PCA_FILE)
    df = pd.read_csv(INPUT_CSV)

    print(f"Issues: {len(df)}")
    print(f"PCA feature shape: {X.shape}")

    if len(df) != len(X):
        raise ValueError(
            "Number of issues and feature rows do not match."
        )

    print("\nRunning Isolation Forest...")

    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42
    )

    model.fit(X)

    anomaly_score = -model.score_samples(X)
    predictions = model.predict(X)

    result = df.copy()

    result["anomaly_score"] = anomaly_score
    result["anomaly"] = predictions == -1

    print("\n" + "=" * 60)
    print("ANOMALY RESULTS")
    print("=" * 60)

    print(
        f"Minimum anomaly score: "
        f"{anomaly_score.min():.4f}"
    )

    print(
        f"Maximum anomaly score: "
        f"{anomaly_score.max():.4f}"
    )

    print(
        f"Mean anomaly score: "
        f"{anomaly_score.mean():.4f}"
    )

    anomaly_count = np.sum(predictions == -1)

    print(
        f"\nDetected anomalies: "
        f"{anomaly_count}"
    )

    print(
        f"Anomaly percentage: "
        f"{anomaly_count / len(predictions) * 100:.2f}%"
    )

    print("\n" + "=" * 60)
    print("MOST ANOMALOUS ISSUES")
    print("=" * 60)

    top_anomalies = (
        result
        .sort_values(
            "anomaly_score",
            ascending=False
        )
        .head(10)
    )

    print(
        top_anomalies[
            [
                "issue_id",
                "technologies",
                "hybrid_concepts",
                "anomaly_score"
            ]
        ].to_string(index=False)
    )

    os.makedirs(
        os.path.dirname(OUTPUT_CSV),
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("\n" + "=" * 60)
    print("COMPLETED")
    print("=" * 60)

    print(
        f"Saved results: {OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()