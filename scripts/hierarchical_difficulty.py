import os
import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


PCA_FILE = "data/difficulty_pca.npy"
INPUT_CSV = "data/issues_with_hybrid_concepts.csv"

OUTPUT_CSV = "data/hierarchical_difficulty.csv"


def main():
    print("Loading PCA features...")

    X = np.load(PCA_FILE)
    df = pd.read_csv(INPUT_CSV)

    print(f"Issues: {len(df)}")
    print(f"PCA feature shape: {X.shape}")

    if len(df) != len(X):
        raise ValueError(
            "Number of issues and PCA feature rows do not match."
        )

    # ---------------------------------------------------------
    # 1. Hierarchical / Agglomerative Clustering
    # ---------------------------------------------------------

    print("\nRunning Hierarchical Clustering...")

    model = AgglomerativeClustering(
        n_clusters=3,
        linkage="ward"
    )

    clusters = model.fit_predict(X)

    # ---------------------------------------------------------
    # 2. Cluster sizes
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("CLUSTER SIZES")
    print("=" * 60)

    cluster_sizes = (
        pd.Series(clusters)
        .value_counts()
        .sort_index()
    )

    print(cluster_sizes)

    # ---------------------------------------------------------
    # 3. Silhouette Score
    # ---------------------------------------------------------

    silhouette = silhouette_score(
        X,
        clusters
    )

    print("\n" + "=" * 60)
    print("SILHOUETTE SCORE")
    print("=" * 60)

    print(
        f"{silhouette:.4f}"
    )

    # ---------------------------------------------------------
    # 4. Save results
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
        f"Saved clustered issues: {OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()