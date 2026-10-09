import os
import csv
import time
import base64
import requests


GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not GITHUB_TOKEN:
    raise ValueError(
        "GITHUB_TOKEN environment variable not found."
    )


HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28"
}

BASE_URL = "https://api.github.com"

TARGET_REPOSITORIES = 500

OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "repositories.csv"
)

SELECTED_FILE = os.path.join(
    OUTPUT_DIR,
    "selected_repositories.csv"
)

SEARCH_QUERIES = [
    "language:Python stars:>10",
    "language:JavaScript stars:>10",
    "language:TypeScript stars:>10",
    "language:Java stars:>10",
    "language:C++ stars:>10",
    "language:C stars:>10",
    "language:C# stars:>10",
    "language:Go stars:>10",
    "language:Rust stars:>10",
    "language:PHP stars:>10",
    "language:Kotlin stars:>10",
    "language:Swift stars:>10",
    "topic:machine-learning stars:>10",
    "topic:deep-learning stars:>10",
    "topic:react stars:>10",
    "topic:django stars:>10",
    "topic:fastapi stars:>10",
    "topic:nodejs stars:>10",
    "topic:vuejs stars:>10",
    "topic:docker stars:>10"
]


FIELDNAMES = [
    "repo_name",
    "full_name",
    "description",
    "html_url",
    "primary_language",
    "languages",
    "topics",
    "stars",
    "forks",
    "open_issues",
    "readme",
    "package_json",
    "requirements_txt",
    "pyproject_toml"
]


def github_get(url, params=None, retries=3):

    for attempt in range(1, retries + 1):

        try:

            response = requests.get(
                url,
                headers=HEADERS,
                params=params,
                timeout=30
            )

            if response.status_code == 200:
                return response.json()

            if response.status_code == 404:
                return None

            if response.status_code == 403:
                print(
                    "GitHub rate limit or temporary block."
                )

            else:
                print(
                    f"Request failed: "
                    f"{response.status_code}"
                )

        except requests.exceptions.RequestException as e:

            print(
                f"Connection error "
                f"(attempt {attempt}/{retries}): {e}"
            )

        if attempt < retries:

            wait_time = 2 ** attempt

            print(
                f"Retrying in "
                f"{wait_time} seconds..."
            )

            time.sleep(wait_time)

    print("Request permanently failed. Skipping.")
    return None


def get_languages(owner, repo):

    url = (
        f"{BASE_URL}/repos/"
        f"{owner}/{repo}/languages"
    )

    data = github_get(url)

    if not data:
        return {}

    return data


def get_topics(owner, repo):

    url = (
        f"{BASE_URL}/repos/"
        f"{owner}/{repo}/topics"
    )

    data = github_get(url)

    if not data:
        return []

    return data.get("names", [])


def get_file_content(
    owner,
    repo,
    filename
):

    url = (
        f"{BASE_URL}/repos/"
        f"{owner}/{repo}/contents/{filename}"
    )

    data = github_get(url)

    if not data:
        return ""

    if isinstance(data, list):
        return ""

    content = data.get("content")

    if not content:
        return ""

    try:

        return base64.b64decode(
            content
        ).decode(
            "utf-8",
            errors="ignore"
        )

    except Exception:

        return ""


def get_readme(owner, repo):

    url = (
        f"{BASE_URL}/repos/"
        f"{owner}/{repo}/readme"
    )

    data = github_get(url)

    if not data:
        return ""

    content = data.get("content")

    if not content:
        return ""

    try:

        return base64.b64decode(
            content
        ).decode(
            "utf-8",
            errors="ignore"
        )

    except Exception:

        return ""


def search_repositories(query):

    repositories = []

    page = 1

    while len(repositories) < 50:

        params = {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": 100,
            "page": page
        }

        url = f"{BASE_URL}/search/repositories"

        data = github_get(
            url,
            params=params
        )

        if not data:
            break

        items = data.get(
            "items",
            []
        )

        if not items:
            break

        repositories.extend(items)

        page += 1

        if len(items) < 100:
            break

        time.sleep(2)

    return repositories[:50]


def select_repositories():

    collected = {}

    print("=" * 60)
    print("Selecting GitHub repositories")
    print("=" * 60)

    for query in SEARCH_QUERIES:

        if len(collected) >= TARGET_REPOSITORIES:
            break

        print()
        print(f"Searching: {query}")

        repositories = search_repositories(
            query
        )

        print(
            f"Found {len(repositories)} repositories"
        )

        for repo in repositories:

            full_name = repo.get(
                "full_name"
            )

            if not full_name:
                continue

            if full_name in collected:
                continue

            if repo.get("fork", False):
                continue

            if repo.get("archived", False):
                continue

            collected[full_name] = repo

            print(
                f"Selected "
                f"{len(collected)}/{TARGET_REPOSITORIES}: "
                f"{full_name}"
            )

            if len(collected) >= TARGET_REPOSITORIES:
                break

        time.sleep(2)

    return list(collected.values())


def save_selected_repositories(repositories):

    with open(
        SELECTED_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "full_name",
                "name",
                "owner"
            ]
        )

        writer.writeheader()

        for repo in repositories:

            writer.writerow({
                "full_name": repo.get(
                    "full_name",
                    ""
                ),
                "name": repo.get(
                    "name",
                    ""
                ),
                "owner": repo.get(
                    "owner",
                    {}
                ).get(
                    "login",
                    ""
                )
            })


def load_completed_repositories():

    completed = set()

    if not os.path.exists(OUTPUT_FILE):
        return completed

    with open(
        OUTPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            full_name = row.get(
                "full_name"
            )

            if full_name:
                completed.add(full_name)

    return completed


def append_repository(row):

    file_exists = os.path.exists(
        OUTPUT_FILE
    )

    with open(
        OUTPUT_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def process_repository(repo):

    full_name = repo["full_name"]

    owner = repo["owner"]["login"]

    repo_name = repo["name"]

    print()
    print(
        f"Processing {full_name}"
    )

    languages = get_languages(
        owner,
        repo_name
    )

    topics = get_topics(
        owner,
        repo_name
    )

    readme = get_readme(
        owner,
        repo_name
    )

    package_json = get_file_content(
        owner,
        repo_name,
        "package.json"
    )

    requirements_txt = get_file_content(
        owner,
        repo_name,
        "requirements.txt"
    )

    pyproject_toml = get_file_content(
        owner,
        repo_name,
        "pyproject.toml"
    )

    row = {
        "repo_name": repo_name,
        "full_name": full_name,
        "description": (
            repo.get("description") or ""
        ),
        "html_url": repo.get(
            "html_url",
            ""
        ),
        "primary_language": (
            repo.get("language") or ""
        ),
        "languages": ",".join(
            languages.keys()
        ),
        "topics": ",".join(
            topics
        ),
        "stars": repo.get(
            "stargazers_count",
            0
        ),
        "forks": repo.get(
            "forks_count",
            0
        ),
        "open_issues": repo.get(
            "open_issues_count",
            0
        ),
        "readme": readme,
        "package_json": package_json,
        "requirements_txt": requirements_txt,
        "pyproject_toml": pyproject_toml
    }

    append_repository(row)


def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    completed = load_completed_repositories()

    print()
    print(
        f"Already completed: "
        f"{len(completed)}"
    )

    repositories = select_repositories()

    save_selected_repositories(
        repositories
    )

    print()
    print(
        f"Total repositories selected: "
        f"{len(repositories)}"
    )

    for index, repo in enumerate(
        repositories,
        start=1
    ):

        full_name = repo["full_name"]

        if full_name in completed:

            print(
                f"[{index}/{len(repositories)}] "
                f"Already completed: "
                f"{full_name}"
            )

            continue

        try:

            process_repository(repo)

            completed.add(full_name)

            print(
                f"Saved "
                f"{len(completed)}/{TARGET_REPOSITORIES}"
            )

        except Exception as e:

            print(
                f"ERROR processing "
                f"{full_name}: {e}"
            )

            print(
                "Skipping this repository."
            )

        time.sleep(0.5)

        if len(completed) >= TARGET_REPOSITORIES:
            break

    print()
    print("=" * 60)
    print("DATASET COLLECTION STATUS")
    print("=" * 60)

    print(
        f"Repositories collected: "
        f"{len(completed)}"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()