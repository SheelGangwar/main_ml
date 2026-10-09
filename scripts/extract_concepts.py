import csv
import os
import re

INPUT_FILE = "data/issues_with_technologies.csv"
OUTPUT_FILE = "data/issues_with_concepts.csv"


CONCEPTS = {
    "authentication": [
        "authentication",
        "auth",
        "login",
        "sign in",
        "signin",
        "jwt",
        "oauth"
    ],

    "authorization": [
        "authorization",
        "permissions",
        "permission",
        "access control",
        "role based access"
    ],

    "api": [
        "api",
        "rest api",
        "rest",
        "graphql",
        "endpoint",
        "endpoints"
    ],

    "database": [
        "database",
        "db",
        "sql",
        "query",
        "queries",
        "schema",
        "migration"
    ],

    "testing": [
        "testing",
        "test",
        "tests",
        "unit test",
        "integration test",
        "pytest",
        "jest"
    ],

    "debugging": [
        "debug",
        "debugging",
        "bug",
        "exception",
        "stack trace",
        "error handling"
    ],

    "performance": [
        "performance",
        "optimization",
        "optimize",
        "slow",
        "latency",
        "memory usage",
        "benchmark"
    ],

    "security": [
        "security",
        "vulnerability",
        "secure",
        "xss",
        "csrf",
        "encryption",
        "password hashing"
    ],

    "caching": [
        "cache",
        "caching",
        "redis",
        "memoization"
    ],

    "deployment": [
        "deployment",
        "deploy",
        "production",
        "release",
        "ci/cd",
        "continuous integration",
        "continuous deployment"
    ],

    "frontend": [
        "frontend",
        "ui",
        "user interface",
        "component",
        "responsive",
        "layout"
    ],

    "backend": [
        "backend",
        "server",
        "server side",
        "middleware"
    ],

    "file handling": [
        "file upload",
        "file handling",
        "filesystem",
        "file system",
        "read file",
        "write file"
    ],

    "data processing": [
        "data processing",
        "data pipeline",
        "data transformation",
        "etl",
        "preprocessing"
    ],

    "machine learning": [
        "machine learning",
        "ml model",
        "training",
        "inference",
        "classification",
        "regression",
        "clustering"
    ],

    "documentation": [
        "documentation",
        "docs",
        "readme",
        "documentation update"
    ],

    "configuration": [
        "configuration",
        "config",
        "environment variable",
        "settings"
    ],

    "logging": [
        "logging",
        "logger",
        "logs",
        "log message"
    ],

    "concurrency": [
        "concurrency",
        "parallel",
        "parallelism",
        "thread",
        "threading",
        "async",
        "asynchronous"
    ]
}


def normalize(text):
    if not text:
        return ""

    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)

    return text


def contains_term(text, term):

    pattern = (
        r"(?<![a-z0-9])"
        + re.escape(term.lower())
        + r"(?![a-z0-9])"
    )

    return re.search(pattern, text) is not None


def extract_concepts(text):

    text = normalize(text)

    found = set()

    for concept, keywords in CONCEPTS.items():

        for keyword in keywords:

            if contains_term(text, keyword):
                found.add(concept)
                break

    return found


def main():

    if not os.path.exists(INPUT_FILE):
        print(f"Input file not found: {INPUT_FILE}")
        return

    processed_rows = []

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            text = " ".join([
                row.get("combined_text", ""),
                row.get("labels", ""),
                row.get("repository_topics", ""),
                row.get("technologies", "")
            ])

            concepts = extract_concepts(text)

            row["concepts"] = "|".join(
                sorted(concepts)
            )

            row["concept_count"] = len(concepts)

            processed_rows.append(row)

    if not processed_rows:
        print("No issues found.")
        return

    fieldnames = list(processed_rows[0].keys())

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(processed_rows)

    print("===================================")
    print("Concept extraction completed!")
    print(f"Processed issues: {len(processed_rows)}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("===================================")


if __name__ == "__main__":
    main()