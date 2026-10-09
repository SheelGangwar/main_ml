import os
import ast
import pandas as pd
import numpy as np
from fastembed import TextEmbedding

INPUT_FILE = "data/issues_with_concepts.csv"
OUTPUT_EMBEDDINGS = "data/issue_embeddings.npy"
OUTPUT_METADATA = "data/issue_embedding_metadata.csv"


def parse_list(value):
    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):
            return [
                str(item).strip()
                for item in parsed
                if str(item).strip()
            ]
    except (ValueError, SyntaxError):
        pass

    return [
        item.strip()
        for item in value.replace("|", ",").split(",")
        if item.strip()
    ]


def build_embedding_text(row):
    parts = []

    title = str(row.get("title", "")).strip()
    description = str(row.get("description", "")).strip()
    language = str(row.get("repository_language", "")).strip()

    technologies = parse_list(row.get("technologies", ""))
    concepts = parse_list(row.get("concepts", ""))
    labels = parse_list(row.get("labels", ""))
    topics = parse_list(row.get("repository_topics", ""))

    if title:
        parts.append(f"Title: {title}")

    if description:
        parts.append(f"Description: {description}")

    if technologies:
        parts.append(
            f"Technologies: {', '.join(technologies)}"
        )

    if concepts:
        parts.append(
            f"Concepts: {', '.join(concepts)}"
        )

    if labels:
        parts.append(
            f"Labels: {', '.join(labels)}"
        )

    if topics:
        parts.append(
            f"Repository topics: {', '.join(topics)}"
        )

    if language:
        parts.append(
            f"Repository language: {language}"
        )

    return ". ".join(parts)


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Input file not found: {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)

    print("===================================")
    print(f"Loaded issues: {len(df)}")
    print("Loading FastEmbed model...")

    model = TextEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    print("Building semantic text...")

    texts = [
        build_embedding_text(row)
        for _, row in df.iterrows()
    ]

    print("Generating embeddings...")

    embeddings = list(
        model.embed(texts)
    )

    embeddings = np.array(embeddings)

    print(f"Embedding shape: {embeddings.shape}")

    os.makedirs("data", exist_ok=True)

    np.save(
        OUTPUT_EMBEDDINGS,
        embeddings
    )

    metadata_columns = [
        "issue_id",
        "issue_number",
        "repo_name",
        "title",
        "description",
        "technologies",
        "concepts",
        "labels",
        "repository_language",
        "repository_topics",
        "state",
        "comments",
        "days_open",
        "label_count",
        "topic_count",
        "title_length",
        "description_length",
        "text_length",
        "technical_term_count",
        "issue_url"
    ]

    available_columns = [
        column
        for column in metadata_columns
        if column in df.columns
    ]

    metadata = df[available_columns].copy()

    metadata.to_csv(
        OUTPUT_METADATA,
        index=False,
        encoding="utf-8"
    )

    print("===================================")
    print("Embedding generation completed!")
    print(f"Embeddings saved to: {OUTPUT_EMBEDDINGS}")
    print(f"Metadata saved to: {OUTPUT_METADATA}")
    print(f"Number of issues: {len(embeddings)}")
    print(f"Embedding dimensions: {embeddings.shape[1]}")
    print("===================================")


if __name__ == "__main__":
    main()