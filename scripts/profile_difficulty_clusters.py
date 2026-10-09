import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

FEATURES_FILE = "data/difficulty_features.npy"
INPUT_CSV = "data/issues_with_hybrid_concepts.csv"

X = np.load(FEATURES_FILE)
df = pd.read_csv(INPUT_CSV)

print("Issues:", len(df))
print("Features:", X.shape)


# --------------------------------------------------
# 1. Train K-Means with 3 clusters
# --------------------------------------------------

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["cluster"] = kmeans.fit_predict(X)


# --------------------------------------------------
# 2. Basic cluster statistics
# --------------------------------------------------

print("\n" + "=" * 60)
print("CLUSTER SIZES")
print("=" * 60)

print(df["cluster"].value_counts().sort_index())


# --------------------------------------------------
# 3. Numeric feature profile
# --------------------------------------------------

numeric_columns = [
    "technical_term_count",
    "technology_count",
    "hybrid_concept_count",
    "description_length",
    "text_length",
    "title_length",
    "topic_count",
    "label_count"
]

print("\n" + "=" * 60)
print("NUMERIC FEATURE PROFILE")
print("=" * 60)

profile = df.groupby("cluster")[numeric_columns].mean().round(2)

print(profile)


# --------------------------------------------------
# 4. Common technologies
# --------------------------------------------------

def get_top_values(series, top_n=10):
    values = []

    for value in series.dropna():
        if str(value).strip():
            values.extend(
                x.strip().lower()
                for x in str(value).split("|")
                if x.strip()
            )

    counts = pd.Series(values).value_counts()

    return counts.head(top_n).to_dict()


print("\n" + "=" * 60)
print("TOP TECHNOLOGIES BY CLUSTER")
print("=" * 60)

for cluster in sorted(df["cluster"].unique()):
    cluster_df = df[df["cluster"] == cluster]

    print(f"\nCluster {cluster}:")
    print(get_top_values(cluster_df["technologies"]))


# --------------------------------------------------
# 5. Common concepts
# --------------------------------------------------

print("\n" + "=" * 60)
print("TOP CONCEPTS BY CLUSTER")
print("=" * 60)

for cluster in sorted(df["cluster"].unique()):
    cluster_df = df[df["cluster"] == cluster]

    print(f"\nCluster {cluster}:")
    print(get_top_values(cluster_df["hybrid_concepts"]))


# --------------------------------------------------
# 6. Common labels
# --------------------------------------------------

print("\n" + "=" * 60)
print("TOP LABELS BY CLUSTER")
print("=" * 60)

for cluster in sorted(df["cluster"].unique()):
    cluster_df = df[df["cluster"] == cluster]

    print(f"\nCluster {cluster}:")
    print(get_top_values(cluster_df["labels"]))


# --------------------------------------------------
# 7. Sample issues
# --------------------------------------------------

print("\n" + "=" * 60)
print("SAMPLE ISSUES")
print("=" * 60)

for cluster in sorted(df["cluster"].unique()):
    print(f"\n--- Cluster {cluster} ---")

    samples = df[df["cluster"] == cluster][
        ["issue_id", "technologies", "hybrid_concepts", "labels"]
    ].head(5)

    print(samples.to_string(index=False))


# --------------------------------------------------
# 8. Save clustered dataset
# --------------------------------------------------

OUTPUT = "data/difficulty_clustered_issues.csv"

df.to_csv(OUTPUT, index=False)

print("\nSaved:", OUTPUT)