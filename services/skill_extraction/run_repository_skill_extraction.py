import ast
import json
import os
import pandas as pd

from .structured_extractor import extract_structured_skills
from .semantic_extractor import extract_semantic_skills
from .evidence_aggregator_ml import aggregate_skill_evidence
from .skill_features_ml import create_ml_features


INPUT_FILE = "data/repositories.csv"

OUTPUT_EVIDENCE_FILE = (
    "data/repository_skill_evidence.csv"
)

OUTPUT_FEATURE_FILE = (
    "data/repository_skill_features.csv"
)


def parse_languages(value):

    if not isinstance(value, str):
        return {}

    value = value.strip()

    if not value:
        return {}

    languages = {}

    for language in value.split(","):

        language = language.strip()

        if language:
            languages[language] = 1

    return languages


def parse_topics(value):

    if not isinstance(value, str):
        return []

    value = value.strip()

    if not value:
        return []

    return [
        topic.strip()
        for topic in value.split(",")
        if topic.strip()
    ]


def parse_package_json(value):

    if not isinstance(value, str):
        return {}

    value = value.strip()

    if not value:
        return {}

    try:
        return json.loads(value)

    except Exception:
        return {}


def parse_requirements(value):

    if not isinstance(value, str):
        return {}

    value = value.strip()

    if not value:
        return {}

    packages = {}

    for line in value.splitlines():

        line = line.strip()

        if not line:
            continue

        if line.startswith("#"):
            continue

        package = (
            line
            .split("==")[0]
            .split(">=")[0]
            .split("<=")[0]
            .split("~=")[0]
            .strip()
        )

        if package:
            packages[package] = ""

    return packages


def parse_pyproject(value):

    if not isinstance(value, str):
        return {}

    value = value.strip()

    if not value:
        return {}

    try:
        import tomllib

        data = tomllib.loads(value)

        return data

    except Exception:
        return {}


def build_repository(row):

    languages = parse_languages(
        row.get("languages", "")
    )

    topics = parse_topics(
        row.get("topics", "")
    )

    package_json = parse_package_json(
        row.get("package_json", "")
    )

    requirements_txt = parse_requirements(
        row.get("requirements_txt", "")
    )

    pyproject_toml = parse_pyproject(
        row.get("pyproject_toml", "")
    )

    return {
        "languages": languages,

        "topics": topics,

        "dependencies": {
            "packageJson": package_json,
            "requirementsTxt": requirements_txt,
            "pyprojectToml": pyproject_toml
        },

        "readme": row.get(
            "readme",
            ""
        )
    }


def main():

    print("=" * 60)
    print("Repository Skill Extraction")
    print("=" * 60)

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Repositories loaded: {len(df)}"
    )

    evidence_rows = []
    feature_rows = []

    for index, row in df.iterrows():

        full_name = row.get(
            "full_name",
            ""
        )

        print()
        print(
            f"[{index + 1}/{len(df)}] "
            f"Processing {full_name}"
        )

        repository = build_repository(
            row
        )

        # -------------------------
        # Structured extraction
        # -------------------------

        structured_results = (
            extract_structured_skills(
                repository
            )
        )

        # -------------------------
        # Semantic extraction
        # -------------------------

        semantic_results = (
            extract_semantic_skills(
                repository.get(
                    "readme",
                    ""
                )
            )
        )

        # -------------------------
        # ML evidence aggregation
        # -------------------------

        aggregated_results = (
            aggregate_skill_evidence(
                structured_results,
                semantic_results
            )
        )

        # -------------------------
        # ML feature creation
        # -------------------------

        features = create_ml_features(
            aggregated_results
        )

        # -------------------------
        # Save repository-level
        # evidence
        # -------------------------

        for result in aggregated_results:

            evidence_rows.append(
                {
                    "full_name": full_name,
                    "skill": result["name"],
                    "evidence": " | ".join(
                        result["evidence"]
                    ),
                    "language_evidence_count":
                        result[
                            "language_evidence_count"
                        ],
                    "dependency_evidence_count":
                        result[
                            "dependency_evidence_count"
                        ],
                    "topic_evidence_count":
                        result[
                            "topic_evidence_count"
                        ],
                    "semantic_evidence_count":
                        result[
                            "semantic_evidence_count"
                        ],
                    "evidence_count":
                        result[
                            "evidence_count"
                        ],
                    "semantic_similarity_max":
                        result[
                            "semantic_similarity_max"
                        ],
                    "semantic_similarity_mean":
                        result[
                            "semantic_similarity_mean"
                        ]
                }
            )

        # -------------------------
        # Save ML features
        # -------------------------

        for feature in features:

            feature_rows.append(
                {
                    "full_name": full_name,
                    **feature
                }
            )

        print(
            f"Skills detected: "
            f"{len(aggregated_results)}"
        )

    # -------------------------
    # Save evidence dataset
    # -------------------------

    evidence_df = pd.DataFrame(
        evidence_rows
    )

    evidence_df.to_csv(
        OUTPUT_EVIDENCE_FILE,
        index=False,
        encoding="utf-8"
    )

    # -------------------------
    # Save feature dataset
    # -------------------------

    feature_df = pd.DataFrame(
        feature_rows
    )

    feature_df.to_csv(
        OUTPUT_FEATURE_FILE,
        index=False,
        encoding="utf-8"
    )

    print()
    print("=" * 60)
    print("SKILL EXTRACTION COMPLETED")
    print("=" * 60)

    print(
        f"Evidence dataset: "
        f"{OUTPUT_EVIDENCE_FILE}"
    )

    print(
        f"Feature dataset: "
        f"{OUTPUT_FEATURE_FILE}"
    )

    print(
        f"Evidence rows: "
        f"{len(evidence_df)}"
    )

    print(
        f"Feature rows: "
        f"{len(feature_df)}"
    )


if __name__ == "__main__":
    main()