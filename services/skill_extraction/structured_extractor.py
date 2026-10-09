from typing import Dict, List, Tuple, Any

from taxonomy import SKILL_TAXONOMY


def normalize(value: str) -> str:
    return value.strip().lower()


def extract_languages(
    languages: Dict[str, int]
) -> List[Tuple[str, str]]:
    """
    Detect programming language skills from GitHub language statistics.
    """

    results = []

    for skill_name, skill_data in SKILL_TAXONOMY.items():

        if skill_data.get("category") != "language":
            continue

        supported_languages = {
            normalize(lang)
            for lang in skill_data.get("langs", [])
        }

        for language in languages:

            if normalize(language) in supported_languages:

                results.append(
                    (
                        skill_name,
                        f"GitHub repository contains {language} code"
                    )
                )

                break

    return results

def extract_topics(
    topics: List[str]
) -> List[Tuple[str, str]]:
    """
    Detect skills from GitHub repository topics.
    """

    results = []

    normalized_topics = {
        normalize(topic)
        for topic in topics
    }

    for skill_name, skill_data in SKILL_TAXONOMY.items():

        skill_topics = {
            normalize(topic)
            for topic in skill_data.get("topics", [])
        }

        matched_topics = normalized_topics.intersection(
            skill_topics
        )

        for topic in matched_topics:

            results.append(
                (
                    skill_name,
                    f"GitHub topic '{topic}' matches this skill"
                )
            )

    return results


def extract_dependencies(
    dependencies: Dict[str, Any]
) -> List[Tuple[str, str]]:
    """
    Detect skills from project dependencies.
    """

    results = []

    package_json = dependencies.get("packageJson") or {}
    requirements_txt = dependencies.get("requirementsTxt") or {}
    pyproject_toml = dependencies.get("pyprojectToml") or {}

    npm_packages = set()

    python_packages = set()

    # -------------------------
    # package.json
    # -------------------------

    if isinstance(package_json, dict):

        for section in [
            "dependencies",
            "devDependencies",
            "peerDependencies"
        ]:

            packages = package_json.get(section, {})

            if isinstance(packages, dict):

                npm_packages.update(
                    normalize(package)
                    for package in packages.keys()
                )

    # -------------------------
    # requirements.txt
    # -------------------------

    if isinstance(requirements_txt, dict):

        python_packages.update(
            normalize(package)
            for package in requirements_txt.keys()
        )

    elif isinstance(requirements_txt, list):

        for package in requirements_txt:

            if isinstance(package, str):

                package_name = (
                    package
                    .split("==")[0]
                    .split(">=")[0]
                    .split("<=")[0]
                    .strip()
                )

                python_packages.add(
                    normalize(package_name)
                )

    # -------------------------
    # pyproject.toml
    # -------------------------

    if isinstance(pyproject_toml, dict):

        for key in [
            "dependencies",
            "devDependencies"
        ]:

            packages = pyproject_toml.get(key, {})

            if isinstance(packages, dict):

                python_packages.update(
                    normalize(package)
                    for package in packages.keys()
                )

            elif isinstance(packages, list):

                for package in packages:

                    if isinstance(package, str):

                        package_name = (
                            package
                            .split("==")[0]
                            .split(">=")[0]
                            .split("<=")[0]
                            .strip()
                        )

                        python_packages.add(
                            normalize(package_name)
                        )

    # -------------------------
    # Match taxonomy
    # -------------------------

    for skill_name, skill_data in SKILL_TAXONOMY.items():

        npm_names = {
            normalize(package)
            for package in skill_data.get("npm", [])
        }

        pypi_names = {
            normalize(package)
            for package in skill_data.get("pypi", [])
        }

        npm_matches = npm_packages.intersection(
            npm_names
        )

        python_matches = python_packages.intersection(
            pypi_names
        )
        
        for package in npm_matches:

            results.append(
                (
                    skill_name,
                    f"Dependency '{package}' found in package.json"
                )
            )

        for package in python_matches:

            results.append(
                (
                    skill_name,
                    f"Python dependency '{package}' found"
                )
            )

    return results


def extract_structured_skills(
    repository: Dict[str, Any]
) -> List[Tuple[str, str]]:
    """
    Run all structured extraction methods.
    """

    results = []

    languages = repository.get(
        "languages",
        {}
    )

    topics = repository.get(
        "topics",
        []
    )

    dependencies = repository.get(
        "dependencies",
        {}
    )

    # Language extraction
    results.extend(
        extract_languages(languages)
    )

    # Topic extraction
    results.extend(
        extract_topics(topics)
    )

    # Dependency extraction
    results.extend(
        extract_dependencies(dependencies)
    )

    return results