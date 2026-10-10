
from functools import lru_cache
from typing import Dict, List, Tuple

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from taxonomy import SKILL_TAXONOMY


MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_THRESHOLD = 0.40


@lru_cache(maxsize=1)
def get_model():
    """Load the transformer model only once per process."""
    return SentenceTransformer(MODEL_NAME)


def build_skill_description(
    skill_name: str,
    skill_data: Dict
) -> str:
    aliases = skill_data.get("aliases", [])
    topics = skill_data.get("topics", [])
    category = skill_data.get("category", "")

    return " ".join([
        skill_name,
        category,
        *aliases,
        *topics
    ])


@lru_cache(maxsize=1)
def get_skill_embeddings():
    """
    Encode taxonomy descriptions once and reuse the embeddings
    across all repository analyses.
    """
    skill_names = list(SKILL_TAXONOMY.keys())

    skill_descriptions = [
        build_skill_description(
            skill_name,
            SKILL_TAXONOMY[skill_name]
        )
        for skill_name in skill_names
    ]

    embeddings = get_model().encode(
        skill_descriptions,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return skill_names, embeddings


def extract_semantic_skills(
    readme: str,
    threshold: float = DEFAULT_THRESHOLD
) -> List[Tuple[str, float, str]]:

    if not readme or not readme.strip():
        return []

    model = get_model()

    skill_names, skill_embeddings = get_skill_embeddings()

    readme_embedding = model.encode(
        [readme],
        normalize_embeddings=True,
        show_progress_bar=False
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
            results.append((
                skill_name,
                float(score),
                f"README semantic similarity with "
                f"{skill_name}: {score:.3f}"
            ))

    results.sort(
        key=lambda item: item[1],
        reverse=True
    )

    return results
