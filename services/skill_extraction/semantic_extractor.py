from typing import Dict, List, Tuple

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from taxonomy import SKILL_TAXONOMY


MODEL_NAME = "all-MiniLM-L6-v2"

_model = None


def get_model():
    """
    Load the sentence transformer model only when needed.
    """

    global _model

    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)

    return _model


def build_skill_description(
    skill_name: str,
    skill_data: Dict
) -> str:
    """
    Create a natural-language description of a skill
    from the taxonomy.
    """

    aliases = skill_data.get("aliases", [])
    topics = skill_data.get("topics", [])
    category = skill_data.get("category", "")

    return " ".join(
        [
            skill_name,
            category,
            *aliases,
            *topics
        ]
    )


def extract_semantic_skills(
    readme: str,
    threshold: float = 0.40
) -> List[Tuple[str, float, str]]:

    if not readme or not readme.strip():
        return []

    model = get_model()

    skill_names = []
    skill_descriptions = []

    for skill_name, skill_data in SKILL_TAXONOMY.items():

        skill_names.append(skill_name)

        skill_descriptions.append(
            build_skill_description(
                skill_name,
                skill_data
            )
        )

    readme_embedding = model.encode(
        [readme],
        normalize_embeddings=True
    )

    skill_embeddings = model.encode(
        skill_descriptions,
        normalize_embeddings=True
    )

    similarities = cosine_similarity(
        readme_embedding,
        skill_embeddings
    )[0]

    results = []

    for skill_name, score in zip(
        skill_names,
        similarities
    ):

        if score >= threshold:

            evidence = (
                f"README semantic similarity "
                f"with {skill_name}: {score:.3f}"
            )

            results.append(
                (
                    skill_name,
                    float(score),
                    evidence
                )
            )

    results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return results