import numpy as np
import pandas as pd
from fastembed import TextEmbedding
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDINGS_FILE = "data/issue_embeddings.npy"

TECHNOLOGIES = [
    "Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "C",
    "Go", "Rust", "PHP", "Ruby", "Kotlin", "Swift", "Scala", "Dart",
    "React", "Angular", "Vue", "Next.js", "Node.js", "Express",
    "Django", "Flask", "FastAPI", "Spring", "Laravel",
    "HTML", "CSS", "Tailwind CSS", "Bootstrap",
    "SQL", "MySQL", "PostgreSQL", "MongoDB", "Redis",
    "Docker", "Kubernetes", "Terraform", "Jenkins",
    "AWS", "Azure", "Google Cloud",
    "TensorFlow", "PyTorch", "Keras", "scikit-learn",
    "Pandas", "NumPy", "Git", "GitHub", "Linux"
]

model = TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

technology_texts = [
    f"{tech} software technology"
    for tech in TECHNOLOGIES
]

technology_embeddings = np.array(
    list(model.embed(technology_texts))
)

issue_embeddings = np.load(EMBEDDINGS_FILE)

similarities = cosine_similarity(
    issue_embeddings,
    technology_embeddings
)

print("Similarity statistics:")
print("Minimum:", similarities.min())
print("Maximum:", similarities.max())
print("Mean:", similarities.mean())

for threshold in [0.45, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:

    counts = (similarities >= threshold).sum(axis=1)

    print(
        f"Threshold {threshold}: "
        f"average matches = {counts.mean():.2f}, "
        f"max matches = {counts.max()}"
    )

print("\nExample issue:")
row = similarities[0]

top_indices = np.argsort(row)[-10:][::-1]

for index in top_indices:
    print(
        TECHNOLOGIES[index],
        round(float(row[index]), 4)
    )