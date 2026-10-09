import pandas as pd
import numpy as np


CLUSTERED_FILE = "data/pca_kmeans_difficulty.csv"


def split_values(value):
    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    return [
        item.strip().lower()
        for item in value.split("|")
        if item.strip()
    ]


def top_values(series, top_n=10):
    counter = {}

    for value in series:
        for item in split_values(value):
            counter[item] = counter.get(item, 0) + 1

    return dict(
        sorted(
            counter.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
    )


def main():

    print("Loading clustered issues...")

    df = pd.read_csv(CLUSTERED_FILE)

    print(f"Issues: {len(df)}")

    if "cluster" not in df.columns:
        raise ValueError(
            "Cluster column not found."
        )

    # ---------------------------------------------------------
    # Cluster sizes
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("CLUSTER SIZES")
    print("=" * 60)

    print(
        df["cluster"]
        .value_counts()
        .sort_index()
    )

    # ---------------------------------------------------------
    # Numeric feature profile
    # ---------------------------------------------------------

    numeric_columns = [
        "technology_count",
        "hybrid_concept_count",
        "technical_term_count",
        "description_length",
        "text_length",
        "label_count",
        "topic_count",
        "comments",
        "days_open",
    ]

    available_numeric = [
        column
        for column in numeric_columns
        if column in df.columns
    ]

    print("\n" + "=" * 60)
    print("NUMERIC FEATURE PROFILE")
    print("=" * 60)

    if available_numeric:

        numeric_profile = (
            df.groupby("cluster")[available_numeric]
            .mean()
            .round(2)
        )

        print(numeric_profile)

    # ---------------------------------------------------------
    # Technologies
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("TOP TECHNOLOGIES BY CLUSTER")
    print("=" * 60)

    if "technologies" in df.columns:

        for cluster in sorted(df["cluster"].unique()):

            cluster_df = df[
                df["cluster"] == cluster
            ]

            print(f"\nCluster {cluster}:")

            print(
                top_values(
                    cluster_df["technologies"],
                    top_n=10
                )
            )

    # ---------------------------------------------------------
    # Concepts
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("TOP CONCEPTS BY CLUSTER")
    print("=" * 60)

    concept_column = None

    if "hybrid_concepts" in df.columns:
        concept_column = "hybrid_concepts"

    elif "concepts" in df.columns:
        concept_column = "concepts"

    if concept_column:

        for cluster in sorted(df["cluster"].unique()):

            cluster_df = df[
                df["cluster"] == cluster
            ]

            print(f"\nCluster {cluster}:")

            print(
                top_values(
                    cluster_df[concept_column],
                    top_n=10
                )
            )

    # ---------------------------------------------------------
    # Labels
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("TOP LABELS BY CLUSTER")
    print("=" * 60)

    if "labels" in df.columns:

        for cluster in sorted(df["cluster"].unique()):

            cluster_df = df[
                df["cluster"] == cluster
            ]

            print(f"\nCluster {cluster}:")

            print(
                top_values(
                    cluster_df["labels"],
                    top_n=10
                )
            )

    # ---------------------------------------------------------
    # Sample issues
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("SAMPLE ISSUES")
    print("=" * 60)

    sample_columns = [
        "issue_id",
        "technologies",
        "hybrid_concepts",
        "labels",
    ]

    available_samples = [
        column
        for column in sample_columns
        if column in df.columns
    ]

    for cluster in sorted(df["cluster"].unique()):

        print(
            f"\n--- Cluster {cluster} ---"
        )

        samples = df[
            df["cluster"] == cluster
        ][available_samples].head(5)

        print(
            samples.to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
    