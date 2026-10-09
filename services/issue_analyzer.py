from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack


BASE_DIR = Path(__file__).resolve().parent.parent 
MODEL_DIR = BASE_DIR / "issue_analyzer_2" / "models" 
technology_artifact = joblib.load( MODEL_DIR / "technology_pipeline.pkl" )
concept_artifact = joblib.load( MODEL_DIR / "concept_pipeline.pkl" )
difficulty_artifact = joblib.load( MODEL_DIR / "difficulty_pipeline.pkl" )


def build_issue_text(
    title="",
    description="",
    labels="",
    repository_language=""
):
    """
    Must match the exact text format used during model training.
    """

    title = str(title or "")
    description = str(description or "")
    labels = str(labels or "")
    repository_language = str(repository_language or "")

    return (
        "TITLE " + title + " "
        "TITLE " + title + " "
        "DESCRIPTION " + description + " "
        "LABELS " + labels + " "
        "LABELS " + labels + " "
        "LANGUAGE " + repository_language
    )


def transform_text(text, artifact):
    """
    Apply the same Word + Character TF-IDF
    vectorizers used during training.
    """

    word_features = artifact["word_vectorizer"].transform([text])
    char_features = artifact["char_vectorizer"].transform([text])

    return hstack(
        [word_features, char_features],
        format="csr"
    )


def predict_multilabel(artifact, text):
    """
    Predict technologies or concepts using:
    - decision scores
    - trained threshold
    - top-k limit
    """

    X = transform_text(text, artifact)

    scores = artifact["model"].decision_function(X)

    # Handle single sample
    scores = scores[0]

    threshold = artifact["threshold"]
    top_k = artifact["top_k"]

    # Select labels above trained threshold
    selected = np.where(scores >= threshold)[0]

    # Keep only top-k highest scoring labels
    if len(selected) > top_k:
        selected = selected[
            np.argsort(scores[selected])[::-1][:top_k]
        ]

    # Fallback: always return the strongest prediction
    if len(selected) == 0:
        selected = np.array(
            [int(np.argmax(scores))]
        )

    # Convert selected indices into binary prediction vector
    prediction = np.zeros(
        (1, len(scores)),
        dtype=int
    )

    prediction[0, selected] = 1

    labels = artifact[
        "label_binarizer"
    ].inverse_transform(prediction)[0]

    return list(labels)


def predict_technology(text):
    return predict_multilabel(
        technology_artifact,
        text
    )


def predict_concepts(text):
    return predict_multilabel(
        concept_artifact,
        text
    )


def predict_difficulty(text):
    X = transform_text(
        text,
        difficulty_artifact
    )

    prediction = difficulty_artifact[
        "model"
    ].predict(X)

    return str(prediction[0])


def analyze_issue(
    title="",
    description="",
    labels="",
    repository_language=""
):
    """
    Complete ML-based issue analysis.

    Output:
    {
        technologies: [...],
        concepts: [...],
        difficulty: "..."
    }
    """

    text = build_issue_text(
        title=title,
        description=description,
        labels=labels,
        repository_language=repository_language
    )

    technologies = predict_technology(text)

    concepts = predict_concepts(text)

    difficulty = predict_difficulty(text)

    return {
        "technologies": technologies,
        "concepts": concepts,
        "difficulty": difficulty
    }
if __name__ == "__main__":
    result = analyze_issue(
        title="Fix React login bug",
        description="The login form does not work correctly on mobile devices.",
        labels="bug,react,frontend",
        repository_language="JavaScript"
    )

    print("\nIssue Analysis Result:")
    print(result)