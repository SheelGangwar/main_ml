from typing import Dict, List


def create_ml_features(
    aggregated_results: List[Dict]
) -> List[Dict]:

    results = []

    for result in aggregated_results:

        features = {
            "skill": result["name"],

            # Evidence presence/count features
            "language_evidence_count": result.get(
                "language_evidence_count", 0
            ),

            "dependency_evidence_count": result.get(
                "dependency_evidence_count", 0
            ),

            "topic_evidence_count": result.get(
                "topic_evidence_count", 0
            ),

            "semantic_evidence_count": result.get(
                "semantic_evidence_count", 0
            ),

            "evidence_count": result.get(
                "evidence_count", 0
            ),

            # Semantic NLP features
            "semantic_similarity_max": result.get(
                "semantic_similarity_max", 0.0
            ),

            "semantic_similarity_mean": result.get(
                "semantic_similarity_mean", 0.0
            )
        }

        # Convert values to numeric types
        for key in features:

            if key == "skill":
                continue

            features[key] = float(
                features[key]
            )

        results.append(features)

    return results