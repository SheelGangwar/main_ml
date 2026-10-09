from typing import Dict, List, Tuple


def aggregate_skill_evidence(
    structured_results: List[Tuple[str, str]],
    semantic_results: List[Tuple[str, float, str]]
) -> List[Dict]:

    skill_data = {}

    # Evidence weights
    evidence_weights = {
        "language": 0.45,
        "dependency": 0.40,
        "topic": 0.20,
    }

    # -----------------------------
    # Structured evidence
    # -----------------------------

    for skill_name, evidence in structured_results:

        if skill_name not in skill_data:
            skill_data[skill_name] = {
                "evidence": [],
                "structured_scores": [],
                "semantic_scores": []
            }

        skill_data[skill_name]["evidence"].append(evidence)

        evidence_lower = evidence.lower()

        if "contains" in evidence_lower and "code" in evidence_lower:
            weight = evidence_weights["language"]

        elif "dependency" in evidence_lower:
            weight = evidence_weights["dependency"]

        elif "topic" in evidence_lower:
            weight = evidence_weights["topic"]

        else:
            weight = 0.10

        skill_data[skill_name]["structured_scores"].append(weight)

    # -----------------------------
    # Semantic evidence
    # -----------------------------

    for skill_name, score, evidence in semantic_results:

        if skill_name not in skill_data:
            skill_data[skill_name] = {
                "evidence": [],
                "structured_scores": [],
                "semantic_scores": []
            }

        skill_data[skill_name]["semantic_scores"].append(score)
        skill_data[skill_name]["evidence"].append(evidence)

    # -----------------------------
    # Final confidence
    # -----------------------------

    results = []

    for skill_name, data in skill_data.items():

        structured_scores = data["structured_scores"]
        semantic_scores = data["semantic_scores"]

        structured_score = min(
            sum(structured_scores),
            1.0
        )

        semantic_score = (
            max(semantic_scores)
            if semantic_scores
            else 0.0
        )

        if structured_score > 0 and semantic_score > 0:

            confidence = (
                0.75 * structured_score
                + 0.25 * semantic_score
            )

        elif structured_score > 0:

            confidence = structured_score

        else:

            confidence = semantic_score

        confidence = min(
            max(confidence, 0.0),
            1.0
        )

        results.append(
            {
                "name": skill_name,
                "confidence": round(confidence, 3),
                "evidence": data["evidence"]
            }
        )

    results.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    return results