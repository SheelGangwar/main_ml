import os
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


FEATURES_FILE = "data/difficulty_features.npy"
INPUT_CSV = "data/issues_with_hybrid_concepts.csv"

OUTPUT_CSV = "data/pca_kmeans_difficulty.csv"
OUTPUT_PCA = "data/difficulty_pca.npy"


def main():
    print("Loading features...")

    X = np.load(FEATURES_FILE)
    df = pd.read_csv(INPUT_CSV)

    print(f"Issues: {len(df)}")
    print(f"Original feature shape: {X.shape}")

    if len(df) != len(X):
        raise ValueError(
            "Number of issues and feature rows do not match."
        )

    # ---------------------------------------------------------
    # 1. PCA
    # ---------------------------------------------------------

    print("\nRunning PCA...")

    # Keep enough components to preserve 90% variance.
    pca = PCA(
        n_components=0.90,
        random_state=42
    )

    X_pca = pca.fit_transform(X)

    print(
        f"PCA feature shape: {X_pca.shape}"
    )

    print(
        f"Explained variance: "
        f"{pca.explained_variance_ratio_.sum():.4f}"
    )

    # Save PCA representation
    np.save(
        OUTPUT_PCA,
        X_pca
    )

    # ---------------------------------------------------------
    # 2. K-Means with k = 3
    # ---------------------------------------------------------

    print("\nRunning K-Means...")

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=20
    )

    clusters = kmeans.fit_predict(X_pca)

    # ---------------------------------------------------------
    # 3. Cluster sizes
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("CLUSTER SIZES")
    print("=" * 60)

    cluster_sizes = pd.Series(
        clusters
    ).value_counts().sort_index()

    print(cluster_sizes)

    # ---------------------------------------------------------
    # 4. Silhouette score
    # ---------------------------------------------------------

    silhouette = silhouette_score(
        X_pca,
        clusters
    )

    print("\n" + "=" * 60)
    print("SILHOUETTE SCORE")
    print("=" * 60)

    print(
        f"{silhouette:.4f}"
    )

    # ---------------------------------------------------------
    # 5. Save results
    # ---------------------------------------------------------

    result = df.copy()

    result["cluster"] = clusters

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
        f"Saved PCA features: {OUTPUT_PCA}"
    )

    print(
        f"Saved clustered issues: {OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()