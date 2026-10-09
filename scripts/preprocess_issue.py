import csv
import re
import os
import ast
from datetime import datetime

INPUT_FILE = "data/cleaned_issues.csv"
OUTPUT_FILE = "data/preprocess_issues.csv"


def normalize_text(text):
    if not text:
        return ""

    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def parse_list(value):
    if not value:
        return []

    value = str(value).strip()

    # Try Python/JSON-like list
    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):
            return [
                str(item).strip().lower()
                for item in parsed
                if str(item).strip()
            ]
    except (ValueError, SyntaxError):
        pass

    # Handle comma-separated values
    if "," in value:
        return [
            item.strip().lower()
            for item in value.split(",")
            if item.strip()
        ]

    # Handle pipe-separated values
    if "|" in value:
        return [
            item.strip().lower()
            for item in value.split("|")
            if item.strip()
        ]

    return [value.lower()] if value else []


def parse_datetime(value):
    if not value:
        return ""

    try:
        dt = datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        )
        return dt.isoformat()
    except ValueError:
        return ""


def calculate_days_open(created_at, state, updated_at):
    if not created_at:
        return 0

    try:
        created = datetime.fromisoformat(
            str(created_at).replace("Z", "+00:00")
        )

        if state == "closed" and updated_at:
            end = datetime.fromisoformat(
                str(updated_at).replace("Z", "+00:00")
            )
        else:
            end = datetime.now(created.tzinfo)

        return max((end - created).days, 0)

    except ValueError:
        return 0


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

            title = normalize_text(row.get("title", ""))
            description = normalize_text(row.get("description", ""))

            labels = parse_list(row.get("labels", ""))
            topics = parse_list(row.get("repository_topics", ""))

            language = normalize_text(
                row.get("repository_language", "")
            )

            state = normalize_text(
                row.get("state", "")
            )

            repo_name = normalize_text(
                row.get("repo_name", "")
            )

            created_at = parse_datetime(
                row.get("created_at", "")
            )

            updated_at = parse_datetime(
                row.get("updated_at", "")
            )

            try:
                issue_number = int(
                    float(row.get("issue_number", 0))
                )
            except (ValueError, TypeError):
                issue_number = 0

            try:
                comments = int(
                    float(row.get("comments", 0))
                )
            except (ValueError, TypeError):
                comments = 0

            combined_text = (
                title + " " + description
            ).strip()

            title_length = len(title)
            description_length = len(description)
            text_length = len(combined_text)

            label_count = len(labels)
            topic_count = len(topics)

            days_open = calculate_days_open(
                created_at,
                state,
                updated_at
            )

            processed_rows.append({
                "issue_id": row.get("issue_id", ""),
                "issue_number": issue_number,
                "repo_name": repo_name,
                "title": title,
                "description": description,
                "combined_text": combined_text,
                "labels": "|".join(labels),
                "label_count": label_count,
                "state": state,
                "repository_language": language,
                "repository_topics": "|".join(topics),
                "topic_count": topic_count,
                "created_at": created_at,
                "updated_at": updated_at,
                "comments": comments,
                "days_open": days_open,
                "title_length": title_length,
                "description_length": description_length,
                "text_length": text_length,
                "issue_url": row.get("issue_url", "")
            })

    fieldnames = [
        "issue_id",
        "issue_number",
        "repo_name",
        "title",
        "description",
        "combined_text",
        "labels",
        "label_count",
        "state",
        "repository_language",
        "repository_topics",
        "topic_count",
        "created_at",
        "updated_at",
        "comments",
        "days_open",
        "title_length",
        "description_length",
        "text_length",
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
        writer.writerows(processed_rows)

    print("===================================")
    print("Preprocessing completed!")
    print(f"Input rows: {len(processed_rows)}")
    print(f"Output file: {OUTPUT_FILE}")
    print("===================================")


if __name__ == "__main__":
    main()