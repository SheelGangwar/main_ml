import csv
import os
import re

INPUT_FILE = "data/preprocess_issues.csv"
OUTPUT_FILE = "data/issue_features.csv"


def count_technical_terms(text):
    if not text:
        return 0

    technical_patterns = [
        r"\bapi\b",
        r"\bdatabase\b",
        r"\bauthentication\b",
        r"\bauthorization\b",
        r"\btesting\b",
        r"\btest\b",
        r"\bbug\b",
        r"\berror\b",
        r"\bexception\b",
        r"\bdeployment\b",
        r"\bperformance\b",
        r"\bsecurity\b",
        r"\brefactor\b",
        r"\boptimization\b",
        r"\bconfiguration\b",
        r"\bintegration\b",
        r"\bbackend\b",
        r"\bfrontend\b",
        r"\balgorithm\b",
        r"\bparser\b",
        r"\bcaching\b"
    ]

    count = 0

    for pattern in technical_patterns:
        count += len(re.findall(pattern, text, re.IGNORECASE))

    return count


def main():

    if not os.path.exists(INPUT_FILE):
        print(f"Input file not found: {INPUT_FILE}")
        return

    rows = []

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            combined_text = row.get("combined_text", "")

            labels = [
                label.strip()
                for label in row.get("labels", "").split("|")
                if label.strip()
            ]

            topics = [
                topic.strip()
                for topic in row.get("repository_topics", "").split("|")
                if topic.strip()
            ]

            try:
                comments = int(row.get("comments", 0))
            except (ValueError, TypeError):
                comments = 0

            try:
                title_length = int(row.get("title_length", 0))
            except (ValueError, TypeError):
                title_length = 0

            try:
                description_length = int(
                    row.get("description_length", 0)
                )
            except (ValueError, TypeError):
                description_length = 0

            try:
                text_length = int(row.get("text_length", 0))
            except (ValueError, TypeError):
                text_length = 0

            try:
                days_open = int(row.get("days_open", 0))
            except (ValueError, TypeError):
                days_open = 0

            feature_row = {
                "issue_id": row.get("issue_id", ""),
                "repo_name": row.get("repo_name", ""),
                "repository_language": row.get(
                    "repository_language", ""
                ),
                "state": row.get("state", ""),

                "label_count": len(labels),
                "topic_count": len(topics),

                "comments": comments,
                "days_open": days_open,

                "title_length": title_length,
                "description_length": description_length,
                "text_length": text_length,

                "technical_term_count":
                    count_technical_terms(combined_text),

                "labels": "|".join(labels),
                "repository_topics": "|".join(topics),

                "combined_text": combined_text,

                "issue_url": row.get("issue_url", "")
            }

            rows.append(feature_row)

    fieldnames = [
        "issue_id",
        "repo_name",
        "repository_language",
        "state",
        "label_count",
        "topic_count",
        "comments",
        "days_open",
        "title_length",
        "description_length",
        "text_length",
        "technical_term_count",
        "labels",
        "repository_topics",
        "combined_text",
        "issue_url"
    ]

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
        writer.writerows(rows)

    print("===================================")
    print("Feature extraction completed!")
    print(f"Input rows: {len(rows)}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("===================================")


if __name__ == "__main__":
    main()