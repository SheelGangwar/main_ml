import os
import re
import numpy as np
import pandas as pd
from fastembed import TextEmbedding
from sklearn.metrics.pairwise import cosine_similarity


EMBEDDINGS_FILE = "data/issue_embeddings.npy"
METADATA_FILE = "data/issue_embedding_metadata.csv"
OUTPUT_FILE = "data/issues_with_hybrid_technologies.csv"


TECHNOLOGIES = {
    "Python": ["python"],
    "JavaScript": ["javascript", "js"],
    "TypeScript": ["typescript", "ts"],
    "Java": ["java"],
    "C++": ["c++", "cpp"],
    "C#": ["c#", "csharp"],
    "C": ["c programming", "c language"],
    "Go": ["golang"],
    "Rust": ["rust"],
    "PHP": ["php"],
    "Ruby": ["ruby"],
    "Kotlin": ["kotlin"],
    "Swift": ["swift"],
    "Scala": ["scala"],
    "Dart": ["dart"],

    "React": ["react", "reactjs", "react.js"],
    "Angular": ["angular"],
    "Vue": ["vue", "vue.js"],
    "Next.js": ["next.js", "nextjs"],
    "Node.js": ["node.js", "nodejs"],
    "Express": ["express", "express.js"],

    "Django": ["django"],
    "Flask": ["flask"],
    "FastAPI": ["fastapi"],
    "Spring": ["spring framework", "spring boot"],
    "Laravel": ["laravel"],
    "Ruby on Rails": ["ruby on rails", "rails"],

    "HTML": ["html"],
    "CSS": ["css"],
    "Sass": ["sass", "scss"],
    "Tailwind CSS": ["tailwind", "tailwind css"],
    "Bootstrap": ["bootstrap"],

    "SQL": ["sql"],
    "MySQL": ["mysql"],
    "PostgreSQL": ["postgresql", "postgres"],
    "MongoDB": ["mongodb", "mongo"],
    "Redis": ["redis"],
    "SQLite": ["sqlite"],
    "Oracle": ["oracle database", "oracle"],

    "Docker": ["docker"],
    "Kubernetes": ["kubernetes", "k8s"],
    "Terraform": ["terraform"],
    "Jenkins": ["jenkins"],
    "GitHub Actions": ["github actions"],

    "AWS": ["aws", "amazon web services"],
    "Azure": ["azure", "microsoft azure"],
    "Google Cloud": ["google cloud", "gcp"],

    "TensorFlow": ["tensorflow"],
    "PyTorch": ["pytorch"],
    "Keras": ["keras"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],

    "Git": ["git"],
    "GitHub": ["github"],
    "Linux": ["linux"]
}


def contains_technology(text, keywords):

    text = str(text).lower()

    for keyword in keywords:
        pattern = r"(?<!\w)" + re.escape(keyword.lower()) + r"(?!\w)"

        if re.search(pattern, text):
            return True

    return False


def main():

    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"Missing: {EMBEDDINGS_FILE}")
        return

    if not os.path.exists(METADATA_FILE):
        print(f"Missing: {METADATA_FILE}")
        return

    issue_embeddings = np.load(EMBEDDINGS_FILE)
    df = pd.read_csv(METADATA_FILE)

    print("===================================")
    print("Hybrid Technology Extraction")
    print("===================================")

    print(f"Issues: {len(df)}")
    print(f"Embedding shape: {issue_embeddings.shape}")

    print("\nLoading FastEmbed...")

    model = TextEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    technology_names = list(TECHNOLOGIES.keys())

    technology_descriptions = [
        f"{technology} software technology"
        for technology in technology_names
    ]

    technology_embeddings = np.array(
        list(model.embed(technology_descriptions))
    )

    similarities = cosine_similarity(
        issue_embeddings,
        technology_embeddings
    )

    results = []

    print("\nExtracting technologies...")

    for row_index in range(len(df)):

        row = df.iloc[row_index]

        text_parts = [
            row.get("repository_language", ""),
            row.get("repository_topics", ""),
            row.get("labels", "")
        ]

        combined_metadata = " ".join(
            str(value) for value in text_parts
        )

        matched = []

        for tech_index, technology in enumerate(technology_names):

            keywords = TECHNOLOGIES[technology]

            explicit_match = contains_technology(
                combined_metadata,
                keywords
            )

            semantic_score = float(
                similarities[row_index][tech_index]
            )

            # Strong evidence:
            # Explicit mention in repository metadata.
            if explicit_match:
                matched.append(technology)
                continue

            # Semantic similarity is only used as
            # supporting evidence.
            if semantic_score >= 0.70:
                matched.append(technology)

        results.append("|".join(sorted(set(matched))))

    df["hybrid_technologies"] = results

    df["hybrid_technology_count"] = (
        df["hybrid_technologies"]
        .apply(
            lambda x: 0 if not x else len(x.split("|"))
        )
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print("\n===================================")
    print("Completed!")
    print("===================================")

    print(
        f"Average technologies per issue: "
        f"{df['hybrid_technology_count'].mean():.2f}"
    )

    print(
        f"Issues with at least one technology: "
        f"{(df['hybrid_technology_count'] > 0).sum()}"
    )

    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
    