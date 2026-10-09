import os
import ast
import numpy as np
import pandas as pd
from fastembed import TextEmbedding
from sklearn.metrics.pairwise import cosine_similarity


EMBEDDINGS_FILE = "data/issue_embeddings.npy"
METADATA_FILE = "data/issue_embedding_metadata.csv"
OUTPUT_FILE = "data/issues_with_semantic_technologies.csv"


TECHNOLOGIES = [
    "Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "C",
    "Go", "Rust", "PHP", "Ruby", "Kotlin", "Swift", "Scala", "Dart",

    "React", "Angular", "Vue", "Next.js", "Node.js", "Express",

    "Django", "Flask", "FastAPI", "Spring", "Laravel", "Ruby on Rails",

    "HTML", "CSS", "Sass", "Tailwind CSS", "Bootstrap",

    "SQL", "MySQL", "PostgreSQL", "MongoDB", "Redis", "SQLite", "Oracle",

    "Docker", "Kubernetes", "Terraform", "Jenkins", "GitHub Actions",

    "AWS", "Azure", "Google Cloud",

    "TensorFlow", "PyTorch", "Keras", "scikit-learn", "Pandas", "NumPy",

    "Git", "GitHub", "Linux"
]


TECHNOLOGY_DESCRIPTIONS = {
    "Python": "Python programming language",
    "JavaScript": "JavaScript programming language",
    "TypeScript": "TypeScript programming language",
    "Java": "Java programming language",
    "C++": "C++ programming language",
    "C#": "C# programming language",
    "C": "C programming language",
    "Go": "Go programming language",
    "Rust": "Rust programming language",
    "PHP": "PHP programming language",
    "Ruby": "Ruby programming language",
    "Kotlin": "Kotlin programming language",
    "Swift": "Swift programming language",
    "Scala": "Scala programming language",
    "Dart": "Dart programming language",

    "React": "React JavaScript frontend library",
    "Angular": "Angular frontend web framework",
    "Vue": "Vue JavaScript frontend framework",
    "Next.js": "Next.js React web framework",
    "Node.js": "Node.js JavaScript runtime",
    "Express": "Express.js Node.js web framework",

    "Django": "Django Python web framework",
    "Flask": "Flask Python web framework",
    "FastAPI": "FastAPI Python web framework",
    "Spring": "Spring Java application framework",
    "Laravel": "Laravel PHP web framework",
    "Ruby on Rails": "Ruby on Rails web framework",

    "HTML": "HTML web markup language",
    "CSS": "CSS web styling language",
    "Sass": "Sass CSS stylesheet language",
    "Tailwind CSS": "Tailwind CSS utility first framework",
    "Bootstrap": "Bootstrap frontend CSS framework",

    "SQL": "SQL relational database language",
    "MySQL": "MySQL relational database",
    "PostgreSQL": "PostgreSQL relational database",
    "MongoDB": "MongoDB NoSQL database",
    "Redis": "Redis in-memory database and caching system",
    "SQLite": "SQLite relational database",
    "Oracle": "Oracle relational database",

    "Docker": "Docker containerization platform",
    "Kubernetes": "Kubernetes container orchestration platform",
    "Terraform": "Terraform infrastructure as code tool",
    "Jenkins": "Jenkins continuous integration tool",
    "GitHub Actions": "GitHub Actions CI/CD automation",

    "AWS": "Amazon Web Services cloud platform",
    "Azure": "Microsoft Azure cloud platform",
    "Google Cloud": "Google Cloud cloud computing platform",

    "TensorFlow": "TensorFlow machine learning framework",
    "PyTorch": "PyTorch machine learning framework",
    "Keras": "Keras deep learning framework",
    "scikit-learn": "scikit-learn machine learning library",
    "Pandas": "Pandas Python data analysis library",
    "NumPy": "NumPy Python numerical computing library",

    "Git": "Git version control system",
    "GitHub": "GitHub software development platform",
    "Linux": "Linux operating system"
}


def parse_list(value):
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, list):
            return parsed
    except:
        pass

    return []


def main():

    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"Embeddings file not found: {EMBEDDINGS_FILE}")
        return

    if not os.path.exists(METADATA_FILE):
        print(f"Metadata file not found: {METADATA_FILE}")
        return

    print("Loading issue embeddings...")

    issue_embeddings = np.load(EMBEDDINGS_FILE)
    df = pd.read_csv(METADATA_FILE)

    print(f"Issues: {len(df)}")
    print(f"Embedding shape: {issue_embeddings.shape}")

    print("\nLoading FastEmbed model...")

    model = TextEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    technology_texts = [
        TECHNOLOGY_DESCRIPTIONS[technology]
        for technology in TECHNOLOGIES
    ]

    print("Generating technology embeddings...")

    technology_embeddings = np.array(
        list(model.embed(technology_texts))
    )

    print(
        f"Technology embedding shape: "
        f"{technology_embeddings.shape}"
    )

    print("\nCalculating semantic similarity...")

    similarities = cosine_similarity(
        issue_embeddings,
        technology_embeddings
    )

    semantic_technologies = []

    threshold = 0.45

    for row in similarities:

        matched = []

        for index, score in enumerate(row):

            if score >= threshold:
                matched.append(
                    (
                        TECHNOLOGIES[index],
                        round(float(score), 4)
                    )
                )

        matched.sort(
            key=lambda x: x[1],
            reverse=True
        )

        semantic_technologies.append(
            [technology for technology, score in matched]
        )

    df["semantic_technologies"] = semantic_technologies

    df["semantic_technology_count"] = (
        df["semantic_technologies"].apply(len)
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print("\n===================================")
    print("Semantic technology extraction completed!")
    print("===================================")

    print(
        f"\nAverage technologies per issue: "
        f"{df['semantic_technology_count'].mean():.2f}"
    )

    print(
        f"\nIssues with at least one technology: "
        f"{(df['semantic_technology_count'] > 0).sum()}"
    )

    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()