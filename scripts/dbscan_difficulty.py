import os
import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


PCA_FILE = "data/difficulty_pca.npy"
INPUT_CSV = "data/issues_with_hybrid_concepts.csv"

OUTPUT_CSV = "data/dbscan_difficulty.csv"


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

    print("\nRunning DBSCAN...")

    model = DBSCAN(
        eps=2.5,
        min_samples=8
    )

    clusters = model.fit_predict(X)

    result = df.copy()
    result["cluster"] = clusters

    print("\n" + "=" * 60)
    print("CLUSTER SIZES")
    print("=" * 60)

    cluster_sizes = (
        pd.Series(clusters)
        .value_counts()
        .sort_index()
    )

    print(cluster_sizes)

    non_noise_mask = clusters != -1

    unique_clusters = set(clusters[non_noise_mask])

    if len(unique_clusters) >= 2:
        silhouette = silhouette_score(
            X[non_noise_mask],
            clusters[non_noise_mask]
        )

        print("\n" + "=" * 60)
        print("SILHOUETTE SCORE")
        print("=" * 60)

        print(f"{silhouette:.4f}")
    else:
        print("\nNot enough clusters for silhouette score.")

    noise_count = np.sum(clusters == -1)

    print("\n" + "=" * 60)
    print("NOISE / OUTLIERS")
    print("=" * 60)

    print(f"Noise points: {noise_count}")
    print(
        f"Noise percentage: "
        f"{noise_count / len(clusters) * 100:.2f}%"
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
        f"Saved clustered issues: {OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()