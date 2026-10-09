import os
import csv
import time
import requests
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN not found in .env file")

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json"
}

REPO = "django/django"
ISSUES_TO_ADD = 50
OUTPUT_FILE = "data/issues.csv"


def get_repo_info():
    url = f"https://api.github.com/repos/{REPO}"

    response = requests.get(
        url,
        headers=HEADERS
    )

    print("Repository API status:", response.status_code)

    if response.status_code != 200:
        print(response.text[:1000])
        return {}

    return response.json()


def get_django_issues():
    collected = []

    for page in range(1, 6):

        url = "https://api.github.com/search/issues"

        params = {
            "q": "repo:django/django is:issue",
            "per_page": 100,
            "page": page
        }

        response = requests.get(
            url,
            headers=HEADERS,
            params=params
        )

        print(f"Search page {page} -> Status {response.status_code}")

        if response.status_code != 200:
            print("GitHub Search API error:")
            print(response.text[:1000])
            break

        data = response.json()

        issues = data.get("items", [])

        print(f"Issues returned on page {page}: {len(issues)}")

        if not issues:
            break

        for issue in issues:

            # Extra safety check:
            # Search query already asks for issues,
            # but don't add anything that has a PR URL.
            if "pull_request" in issue:
                continue

            collected.append(issue)

            if len(collected) >= ISSUES_TO_ADD:
                return collected

        time.sleep(1)

    return collected


def main():

    print("===================================")
    print("Django Issue Collector")
    print("===================================")

    repo_info = get_repo_info()

    if not repo_info:
        return

    print("Repository:", REPO)
    print("Language:", repo_info.get("language"))

    print("\nSearching for Django issues...")

    issues = get_django_issues()

    print("\n===================================")
    print(f"Actual Django issues collected: {len(issues)}")
    print("===================================")

    if not issues:
        print("No Django issues found.")
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

    os.makedirs("data", exist_ok=True)

    file_exists = os.path.exists(OUTPUT_FILE)

    with open(
        OUTPUT_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        if not file_exists or os.path.getsize(OUTPUT_FILE) == 0:
            writer.writeheader()

        for issue in issues:

            labels = [
                label.get("name", "")
                for label in issue.get("labels", [])
            ]

            writer.writerow({
                "issue_id": issue.get("id"),
                "issue_number": issue.get("number"),
                "repo_name": REPO,
                "title": issue.get("title", ""),
                "description": issue.get("body") or "",
                "labels": ", ".join(labels),
                "state": issue.get("state", ""),
                "repository_language": repo_info.get("language", ""),
                "repository_topics": ", ".join(
                    repo_info.get("topics", [])
                ),
                "created_at": issue.get("created_at", ""),
                "updated_at": issue.get("updated_at", ""),
                "comments": issue.get("comments", 0),
                "issue_url": issue.get("html_url", "")
            })

    print("\nDjango issues added successfully!")
    print(f"Added: {len(issues)}")
    print(f"Updated file: {OUTPUT_FILE}")
    print("===================================")


if __name__ == "__main__":
    main()