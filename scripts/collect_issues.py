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

REPOSITORIES = [
    "scikit-learn/scikit-learn",
    "pallets/flask",
    "django/django",
    "fastapi/fastapi",
    "facebook/react",
    "nodejs/node",
    "vuejs/core",
    "spring-projects/spring-boot",
    "tensorflow/tensorflow",
    "pytorch/pytorch",
    "numpy/numpy",
    "pandas-dev/pandas",
    "microsoft/vscode",
    "docker/compose",
    "kubernetes/kubernetes",
    "rust-lang/rust",
    "golang/go",
    "rails/rails",
    "laravel/framework",
    "flutter/flutter"
]

OUTPUT_FILE = "data/issues.csv"
ISSUES_PER_REPO = 50


def get_repository_info(repo):
    url = f"https://api.github.com/repos/{repo}"

    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        print(f"Could not get repository info: {repo}")
        print(f"Status: {response.status_code}")
        return {}

    return response.json()


def get_issues(repo):
    all_issues = []

    for page in range(1, 6):  # maximum 5 pages
        url = f"https://api.github.com/repos/{repo}/issues"

        params = {
            "state": "all",
            "per_page": 100,
            "page": page
        }

        response = requests.get(
            url,
            headers=HEADERS,
            params=params
        )

        if response.status_code != 200:
            print(f"Could not get issues: {repo}")
            print(f"Status: {response.status_code}")
            break

        issues = response.json()

        if not issues:
            break

        for issue in issues:
            # Ignore pull requests
            if "pull_request" in issue:
                continue

            all_issues.append(issue)

            if len(all_issues) >= ISSUES_PER_REPO:
                return all_issues

        time.sleep(0.5)

    return all_issues


def main():

    all_issues = []

    print("Starting GitHub issue collection...\n")

    for repo in REPOSITORIES:

        print(f"Collecting: {repo}")

        repo_info = get_repository_info(repo)

        if not repo_info:
            continue

        repository_language = repo_info.get("language")
        repository_topics = repo_info.get("topics", [])

        issues = get_issues(repo)

        count = 0

        for issue in issues:

            # GitHub API returns pull requests along with issues.
            # Ignore pull requests.
            if "pull_request" in issue:
                continue

            labels = [
                label.get("name", "")
                for label in issue.get("labels", [])
            ]

            issue_data = {
                "issue_id": issue.get("id"),
                "issue_number": issue.get("number"),
                "repo_name": repo,
                "title": issue.get("title", ""),
                "description": issue.get("body") or "",
                "labels": ", ".join(labels),
                "state": issue.get("state", ""),
                "repository_language": repository_language or "",
                "repository_topics": ", ".join(repository_topics),
                "created_at": issue.get("created_at", ""),
                "updated_at": issue.get("updated_at", ""),
                "comments": issue.get("comments", 0),
                "issue_url": issue.get("html_url", "")
            }

            all_issues.append(issue_data)

            count += 1

            if count >= ISSUES_PER_REPO:
                break

        print(f"Collected {count} issues\n")

        time.sleep(1)

    os.makedirs("data", exist_ok=True)

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
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(all_issues)

    print("===================================")
    print("Dataset collection completed!")
    print(f"Total issues collected: {len(all_issues)}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("===================================")


if __name__ == "__main__":
    main()