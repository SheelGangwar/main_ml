import pandas as pd
import numpy as np
from scipy.stats import spearmanr


INPUT_FILE = "data/isolation_forest_difficulty.csv"


def main():
    print("Loading anomaly results...")

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "anomaly_score",
        "technologies",
        "hybrid_concepts"
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Missing column: {column}"
            )

    # Count technologies and concepts
    df["technology_count"] = (
        df["technologies"]
        .fillna("")
        .apply(
            lambda x: len(
                [item for item in str(x).split("|") if item.strip()]
            )
        )
    )

    df["concept_count"] = (
        df["hybrid_concepts"]
        .fillna("")
        .apply(
            lambda x: len(
                [item for item in str(x).split("|") if item.strip()]
            )
        )
    )

    print("\n" + "=" * 60)
    print("ANOMALY SCORE STATISTICS")
    print("=" * 60)

    print(
        df["anomaly_score"].describe()
    )

    print("\n" + "=" * 60)
    print("CORRELATION WITH COMPLEXITY FEATURES")
    print("=" * 60)

    correlations = {}

    for feature in [
        "technology_count",
        "concept_count"
    ]:
        correlation, p_value = spearmanr(
            df["anomaly_score"],
            df[feature]
        )

        correlations[feature] = correlation

        print(
            f"{feature}: "
            f"correlation={correlation:.4f}, "
            f"p-value={p_value:.6f}"
        )

    print("\n" + "=" * 60)
    print("AVERAGE FEATURES BY ANOMALY STATUS")
    print("=" * 60)

    summary = (
        df.groupby("anomaly")[
            [
                "anomaly_score",
                "technology_count",
                "concept_count"
            ]
        ]
        .mean()
    )

    print(summary)

    print("\n" + "=" * 60)
    print("TOP 15 ANOMALOUS ISSUES")
    print("=" * 60)

    top = (
        df.sort_values(
            "anomaly_score",
            ascending=False
        )
        .head(15)
    )

    print(
        top[
            [
                "issue_id",
                "technology_count",
                "concept_count",
                "anomaly_score"
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 60)
    print("COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()