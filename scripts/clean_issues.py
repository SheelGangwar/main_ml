import csv
import re
import os

INPUT_FILE = "data/issues.csv"
OUTPUT_FILE = "data/cleaned_issues.csv"


def clean_text(text):
    if not text:
        return ""

    

    # Remove inline code backticks
    text = re.sub(r"`([^`]*)`", r"\1", text)

    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove markdown images
    text = re.sub(r"!\[.*?\]\(.*?\)", " ", text)

    # Convert markdown links to their visible text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Remove markdown headings
    text = re.sub(r"#{1,6}\s*", " ", text)

    # Remove markdown emphasis symbols
    text = re.sub(r"[*_~]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


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

            title = clean_text(row.get("title", ""))
            description = clean_text(row.get("description", ""))

            # Skip issues with no useful text
            if not title and not description:
                continue

            row["title"] = title
            row["description"] = description

            rows.append(row)

    # Remove duplicate issues using issue_id
    unique_rows = []
    seen_ids = set()

    for row in rows:

        issue_id = row.get("issue_id", "")

        if issue_id in seen_ids:
            continue

        seen_ids.add(issue_id)
        unique_rows.append(row)

    if not unique_rows:
        print("No usable issues found.")
        return

    fieldnames = [
        "issue_id",
        "issue_number",
        "repo_name",
        "title",
        "description",
        "labels",
        "state",
        "repository_language",
        "repository_topics",
        "created_at",
        "updated_at",
        "comments",
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
        writer.writerows(unique_rows)

    print("===================================")
    print("Cleaning completed!")
    print(f"Original rows: {len(rows)}")
    print(f"Unique usable issues: {len(unique_rows)}")
    print(f"Removed duplicates: {len(rows) - len(unique_rows)}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("===================================")


if __name__ == "__main__":
    main()