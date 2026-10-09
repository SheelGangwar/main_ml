import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

FEATURES_FILE = "data/difficulty_features.npy"

X = np.load(FEATURES_FILE)

print("Feature shape:", X.shape)

results = []

for k in range(2, 7):
    print(f"\nTesting k={k}...")

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    cluster_labels = model.fit_predict(X)

    score = silhouette_score(X, cluster_labels)

    cluster_sizes = np.bincount(cluster_labels)

    print("Silhouette Score:", round(score, 4))
    print("Cluster sizes:", cluster_sizes.tolist())

    results.append((k, score, cluster_sizes))


print("\n" + "=" * 50)
print("CLUSTERING SUMMARY")
print("=" * 50)

for k, score, sizes in results:
    print(
        f"k={k} | "
        f"Silhouette={score:.4f} | "
        f"Sizes={sizes.tolist()}"
    )