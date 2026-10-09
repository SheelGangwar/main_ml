
import pandas as pd
import numpy as np
from fastembed import TextEmbedding


INPUT_FILE = "data/issues_with_concepts.csv"
EMBEDDINGS_FILE = "data/issue_embeddings.npy"
METADATA_FILE = "data/issue_embedding_metadata.csv"
OUTPUT_FILE = "data/issues_with_hybrid_concepts.csv"


# Broader concept vocabulary
CONCEPT_ALIASES = {
    "API": [
        "api", "rest api", "restful api", "endpoint", "endpoints"
    ],
    "authentication": [
        "authentication", "authenticate", "login", "logout",
        "sign in", "sign up", "oauth", "jwt"
    ],
    "authorization": [
        "authorization", "authorize", "permissions",
        "permission", "access control", "role based access"
    ],
    "database": [
        "database", "databases", "db", "sql", "query",
        "queries", "schema", "migration"
    ],
    "testing": [
        "testing", "test", "tests", "unit test",
        "integration test", "test case", "test suite"
    ],
    "debugging": [
        "debugging", "debug", "bug", "bugs",
        "error", "errors", "exception", "exceptions",
        "crash", "failure"
    ],
    "performance": [
        "performance", "slow", "latency", "throughput",
        "optimization", "optimize", "bottleneck"
    ],
    "security": [
        "security", "secure", "vulnerability",
        "vulnerabilities", "exploit", "encryption"
    ],
    "caching": [
        "cache", "caching", "cached"
    ],
    "deployment": [
        "deployment", "deploy", "deployed",
        "production", "release", "hosting"
    ],
    "frontend": [
        "frontend", "front-end", "ui", "user interface",
        "web interface", "component"
    ],
    "backend": [
        "backend", "back-end", "server", "server-side"
    ],
    "file handling": [
        "file handling", "file upload", "file download",
        "file processing", "filesystem", "file system"
    ],
    "data processing": [
        "data processing", "data preprocessing",
        "data pipeline", "data transformation",
        "data cleaning"
    ],
    "machine learning": [
        "machine learning", "machine-learning",
        "ml", "model training", "training model",
        "prediction", "classifier", "classification"
    ],
    "data analysis": [
        "data analysis", "data analytics",
        "data analysis", "statistical analysis"
    ],
    "documentation": [
        "documentation", "document", "docs",
        "readme", "read me", "guide", "tutorial"
    ],
    "configuration": [
        "configuration", "config", "settings",
        "environment variables", "env variable"
    ],
    "logging": [
        "logging", "logger", "log", "logs"
    ],
    "concurrency": [
        "concurrency", "concurrent", "parallel",
        "parallelism", "multithreading", "threading",
        "threads", "async", "asynchronous"
    ],
    "networking": [
        "network", "networking", "http", "https",
        "request", "requests", "socket"
    ],
    "version control": [
        "git", "github", "branch", "branches",
        "commit", "commits", "merge", "pull request"
    ],
    "dependency management": [
        "dependency", "dependencies", "package",
        "packages", "requirements", "version conflict"
    ],
    "build system": [
        "build", "build system", "compile", "compilation",
        "compiler", "build process"
    ],
    "refactoring": [
        "refactor", "refactoring", "cleanup",
        "code cleanup", "restructure"
    ],
    "bug fixing": [
        "bug fix", "bugfix", "fix bug",
        "fix issue", "fix error"
    ],
    "user interface": [
        "user interface", "interface", "ui",
        "ux", "user experience"
    ],
    "architecture": [
        "architecture", "design pattern",
        "system design", "software design"
    ],
    "automation": [
        "automation", "automate", "automated",
        "workflow", "pipeline"
    ],
    "monitoring": [
        "monitoring", "monitor", "metrics",
        "observability", "health check"
    ],
    "serialization": [
        "serialization", "deserialization",
        "json", "xml", "yaml"
    ],
    "memory management": [
        "memory management", "memory leak",
        "memory usage", "garbage collection"
    ]
}


def normalize_text(value):
    if pd.isna(value):
        return ""

    return str(value).lower()


def normalize_concept_name(concept):
    """
    Converts different spellings/cases to one canonical concept.
    """

    concept = concept.strip().lower()

    for canonical, aliases in CONCEPT_ALIASES.items():
        if concept == canonical.lower():
            return canonical

        if concept in [alias.lower() for alias in aliases]:
            return canonical

    return concept


def build_issue_text(row):
    """
    Combine all available issue information.
    """

    fields = [
        "title",
        "description",
        "labels",
        "repository_topics",
        "repository_language",
        "technologies",
        "concepts"
    ]

    text_parts = []

    for field in fields:
        if field in row.index:
            value = normalize_text(row[field])

            if value:
                text_parts.append(value)

    return " ".join(text_parts)


def explicit_concept_match(concept, text):
    """
    Detect concept using explicit aliases.
    """

    aliases = CONCEPT_ALIASES[concept]

    for alias in aliases:
        alias = alias.lower()

        if alias in text:
            return True

    return False


def cosine_similarity(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0 or b_norm == 0:
        return 0.0

    return float(np.dot(a, b) / (a_norm * b_norm))


print("Loading data...")

df = pd.read_csv(INPUT_FILE)

issue_embeddings = np.load(EMBEDDINGS_FILE)

metadata = pd.read_csv(METADATA_FILE)

print(f"Issues: {len(df)}")
print(f"Embedding shape: {issue_embeddings.shape}")
print(f"Concept vocabulary size: {len(CONCEPT_ALIASES)}")


print("\nLoading FastEmbed...")

model = TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)


# Create semantic representations for concepts
concept_names = list(CONCEPT_ALIASES.keys())

concept_texts = [
    f"{concept} software development concept"
    for concept in concept_names
]

concept_embeddings = np.array(
    list(model.embed(concept_texts))
)


print("\nExtracting hybrid concepts...")


results = []

for i, row in df.iterrows():

    selected_concepts = set()

    # --------------------------------------------------
    # 1. Preserve existing extracted concepts
    # --------------------------------------------------

    existing_concepts = normalize_text(
        row.get("concepts", "")
    )

    if existing_concepts:

        for concept in existing_concepts.split("|"):

            concept = concept.strip()

            if concept:

                canonical = normalize_concept_name(
                    concept
                )

                selected_concepts.add(canonical)


    # --------------------------------------------------
    # 2. Explicit concept detection
    # --------------------------------------------------

    issue_text = build_issue_text(row)

    for concept in concept_names:

        if explicit_concept_match(
            concept,
            issue_text
        ):

            selected_concepts.add(concept)


    # --------------------------------------------------
    # 3. Semantic similarity
    # --------------------------------------------------

    issue_vector = issue_embeddings[i]

    semantic_scores = []

    for j, concept in enumerate(concept_names):

        score = cosine_similarity(
            issue_vector,
            concept_embeddings[j]
        )

        semantic_scores.append(
            (concept, score)
        )


    # --------------------------------------------------
    # 4. Add only strong semantic matches
    # --------------------------------------------------

    # Slightly stricter threshold to reduce false positives
    SEMANTIC_THRESHOLD = 0.72

    for concept, score in semantic_scores:

        if score >= SEMANTIC_THRESHOLD:

            selected_concepts.add(concept)


    # --------------------------------------------------
    # 5. Canonical normalization
    # --------------------------------------------------

    normalized_concepts = set()

    for concept in selected_concepts:

        canonical = normalize_concept_name(
            concept
        )

        if canonical:

            normalized_concepts.add(
                canonical
            )


    # Sort for consistent output
    final_concepts = sorted(
        normalized_concepts
    )

    results.append(
        "|".join(final_concepts)
    )


# ------------------------------------------------------
# Save results
# ------------------------------------------------------

df["hybrid_concepts"] = results

df["hybrid_concept_count"] = (
    df["hybrid_concepts"]
    .apply(
        lambda x:
        len(x.split("|"))
        if x
        else 0
    )
)


df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n===================================")
print("Completed!")
print("===================================")

print(
    "Concept vocabulary:",
    len(CONCEPT_ALIASES)
)

print(
    "Average concepts per issue:",
    round(
        df["hybrid_concept_count"].mean(),
        2
    )
)

print(
    "Issues with at least one concept:",
    (
        df["hybrid_concept_count"] > 0
    ).sum()
)

print(
    "Issues with no concept:",
    (
        df["hybrid_concept_count"] == 0
    ).sum()
)

print(
    f"Saved to: {OUTPUT_FILE}"
)

