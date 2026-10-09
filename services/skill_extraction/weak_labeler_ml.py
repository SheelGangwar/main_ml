from typing import Dict, Any

from ...taxonomy import SKILL_TAXONOMY


def generate_weak_label(
    features: Dict[str, Any]
) -> Dict[str, Any]:

    skill = features.get("skill", "")

    language_count = int(
        features.get(
            "language_evidence_count",
            0
        )
    )

    dependency_count = int(
        features.get(
            "dependency_evidence_count",
            0
        )
    )

    topic_count = int(
        features.get(
            "topic_evidence_count",
            0
        )
    )

    semantic_count = int(
        features.get(
            "semantic_evidence_count",
            0
        )
    )

    evidence_count = int(
        features.get(
            "evidence_count",
            0
        )
    )

    semantic_similarity = float(
        features.get(
            "semantic_similarity_max",
            0.0
        )
    )

    reasons = []

    skill_data = SKILL_TAXONOMY.get(
        skill,
        {}
    )

    category = skill_data.get(
        "category",
        ""
    )

    # ========================================
    # STEP 1: Strong direct evidence
    # ========================================

    # Language evidence is strong ONLY
    # when the skill itself is a programming language.

    if (
        category == "language"
        and language_count > 0
    ):

        reasons.append(
            "direct language evidence"
        )

        return {
            "weak_label": 1,
            "label_type": "positive",
            "label_source": "deterministic",
            "label_confidence": 0.99,
            "label_reasons": reasons,
            "positive_signal_count": 1
        }

    # Direct dependency evidence is strong
    # for frameworks, libraries, tools, etc.

    if dependency_count > 0:

        reasons.append(
            "direct dependency evidence"
        )

        return {
            "weak_label": 1,
            "label_type": "positive",
            "label_source": "deterministic",
            "label_confidence": 0.99,
            "label_reasons": reasons,
            "positive_signal_count": 1
        }

    # ========================================
    # STEP 2: Multiple independent signals
    # ========================================

    if topic_count > 0:

        reasons.append(
            "topic evidence"
        )

    if semantic_similarity >= 0.50:

        reasons.append(
            "strong semantic similarity"
        )

    independent_signals = 0

    if topic_count > 0:
        independent_signals += 1

    if semantic_similarity >= 0.50:
        independent_signals += 1

    if (
        independent_signals >= 2
    ):

        return {
            "weak_label": 1,
            "label_type": "positive",
            "label_source": "deterministic",
            "label_confidence": 0.90,
            "label_reasons": reasons,
            "positive_signal_count":
                independent_signals
        }

    # ========================================
    # STEP 3: Ambiguous
    # ========================================

    if (
        topic_count > 0
        or semantic_count > 0
        or evidence_count > 0
    ):

        return {
            "weak_label": None,
            "label_type": "uncertain",
            "label_source": "pending",
            "label_confidence": 0.0,
            "label_reasons": reasons,
            "positive_signal_count":
                independent_signals
        }

    # ========================================
    # STEP 4: No evidence
    # ========================================

    # Do NOT call this negative.
    # Absence of evidence is not enough.

    return {
        "weak_label": None,
        "label_type": "uncertain",
        "label_source": "pending",
        "label_confidence": 0.0,
        "label_reasons": [],
        "positive_signal_count": 0
    }