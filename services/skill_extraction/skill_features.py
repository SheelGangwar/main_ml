from typing import Dict, List
import re


def _create_empty_features(skill_name: str) -> Dict:
    return {
        "skill": skill_name,

        # Repository-level signals
        "repo_count": 0,
        "skill_repo_count": 0,
        "repository_coverage": 0.0,

        # Evidence signals
        "language_evidence_count": 0,
        "dependency_evidence_count": 0,
        "topic_evidence_count": 0,
        "semantic_evidence_count": 0,
        "evidence_count": 0,

        # Semantic NLP signals
        "semantic_similarity_max": 0.0,
        "semantic_similarity_mean": 0.0,

        # Detection confidence signals
        "confidence_max": 0.0,
        "confidence_mean": 0.0,
    }


def _analyse_evidence(
    evidence: List[str]
) -> Dict:

    language_count = 0
    dependency_count = 0
    topic_count = 0
    semantic_count = 0

    semantic_scores = []

    for item in evidence:

        text = item.lower()

        # Language evidence
        if "contains" in text and "code" in text:
            language_count += 1

        # Dependency evidence
        elif "dependency" in text:
            dependency_count += 1

        # Topic evidence
        elif "topic" in text:
            topic_count += 1

        # Semantic README evidence
        elif "semantic similarity" in text:

            semantic_count += 1

            match = re.search(
                r":\s*([0-9]*\.?[0-9]+)\s*$",
                item
            )

            if match:
                semantic_scores.append(
                    float(match.group(1))
                )

    return {
        "language_evidence_count": language_count,
        "dependency_evidence_count": dependency_count,
        "topic_evidence_count": topic_count,
        "semantic_evidence_count": semantic_count,
        "semantic_scores": semantic_scores,
        "evidence_count": len(evidence)
    }


def build_student_skill_features(
    all_repo_results: List[List[Dict]]
) -> List[Dict]:
    """
    Convert repository-wise skill extraction results
    into student-level skill feature vectors.

    all_repo_results example:

    [
        [
            {
                "name": "React",
                "confidence": 0.56,
                "evidence": [...]
            }
        ],
        [
            {
                "name": "React",
                "confidence": 0.72,
                "evidence": [...]
            }
        ]
    ]
    """

    repo_count = len(all_repo_results)

    skill_data: Dict[str, Dict] = {}

    # -----------------------------------------
    # Process every repository
    # -----------------------------------------

    for repo_results in all_repo_results:

        # Keep track of which skills appeared
        # in this repository.
        skills_in_repo = set()

        for result in repo_results:

            skill_name = result["name"]

            confidence = float(
                result.get("confidence", 0.0)
            )

            evidence = result.get(
                "evidence",
                []
            )

            if skill_name not in skill_data:

                skill_data[skill_name] = {
                    "repo_count": 0,
                    "confidences": [],
                    "evidence": [],
                    "language_count": 0,
                    "dependency_count": 0,
                    "topic_count": 0,
                    "semantic_count": 0,
                    "semantic_scores": []
                }

            # Count this repository only once
            # for a particular skill.
            if skill_name not in skills_in_repo:

                skill_data[skill_name]["repo_count"] += 1

                skills_in_repo.add(
                    skill_name
                )

            skill_data[skill_name][
                "confidences"
            ].append(confidence)

            skill_data[skill_name][
                "evidence"
            ].extend(evidence)

            evidence_data = _analyse_evidence(
                evidence
            )

            skill_data[skill_name][
                "language_count"
            ] += evidence_data[
                "language_evidence_count"
            ]

            skill_data[skill_name][
                "dependency_count"
            ] += evidence_data[
                "dependency_evidence_count"
            ]

            skill_data[skill_name][
                "topic_count"
            ] += evidence_data[
                "topic_evidence_count"
            ]

            skill_data[skill_name][
                "semantic_count"
            ] += evidence_data[
                "semantic_evidence_count"
            ]

            skill_data[skill_name][
                "semantic_scores"
            ].extend(
                evidence_data[
                    "semantic_scores"
                ]
            )

    # -----------------------------------------
    # Build final feature vectors
    # -----------------------------------------

    results = []

    for skill_name, data in skill_data.items():

        features = _create_empty_features(
            skill_name
        )

        skill_repo_count = data[
            "repo_count"
        ]

        confidences = data[
            "confidences"
        ]

        semantic_scores = data[
            "semantic_scores"
        ]

        evidence = data[
            "evidence"
        ]

        features["repo_count"] = repo_count

        features[
            "skill_repo_count"
        ] = skill_repo_count

        if repo_count > 0:

            features[
                "repository_coverage"
            ] = round(
                skill_repo_count / repo_count,
                3
            )

        features[
            "language_evidence_count"
        ] = data["language_count"]

        features[
            "dependency_evidence_count"
        ] = data["dependency_count"]

        features[
            "topic_evidence_count"
        ] = data["topic_count"]

        features[
            "semantic_evidence_count"
        ] = data["semantic_count"]

        features[
            "evidence_count"
        ] = len(evidence)

        # Semantic similarity
        if semantic_scores:

            features[
                "semantic_similarity_max"
            ] = round(
                max(semantic_scores),
                3
            )

            features[
                "semantic_similarity_mean"
            ] = round(
                sum(semantic_scores)
                / len(semantic_scores),
                3
            )

        # Detection confidence
        if confidences:

            features[
                "confidence_max"
            ] = round(
                max(confidences),
                3
            )

            features[
                "confidence_mean"
            ] = round(
                sum(confidences)
                / len(confidences),
                3
            )

        results.append(features)

    # Strongest skills first
    results.sort(
        key=lambda x: x["confidence_max"],
        reverse=True
    )

    return results