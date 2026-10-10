from typing import Dict, List, Tuple, Any
import re

from taxonomy import SKILL_TAXONOMY


def normalize(value: str) -> str:
    return value.strip().lower()


# =========================================================
# LANGUAGES
# =========================================================

def extract_languages(
    languages: Dict[str, int]
) -> List[Tuple[str, str]]:

    results = []

    if not isinstance(languages, dict):
        return results

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


# =========================================================
# GITHUB TOPICS
# =========================================================

def extract_topics(
    topics: List[str]
) -> List[Tuple[str, str]]:

    results = []

    if not isinstance(topics, list):
        return results

    normalized_topics = {
        normalize(topic)
        for topic in topics
        if isinstance(topic, str)
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


# =========================================================
# PACKAGE.JSON
# =========================================================

def extract_npm_packages(
    package_files: Any
) -> set:

    packages_found = set()

    if not isinstance(package_files, list):
        return packages_found

    for file_info in package_files:

        if not isinstance(file_info, dict):
            continue

        package_json = file_info.get("data")

        if not isinstance(package_json, dict):
            continue

        for section in [
            "dependencies",
            "devDependencies",
            "peerDependencies",
            "optionalDependencies",
        ]:

            packages = package_json.get(
                section,
                {}
            )

            if isinstance(packages, dict):

                for package_name in packages.keys():

                    packages_found.add(
                        normalize(package_name)
                    )

    return packages_found


# =========================================================
# REQUIREMENTS.TXT
# =========================================================

def extract_python_requirements(
    requirement_files: Any
) -> set:

    packages_found = set()

    if not isinstance(requirement_files, list):
        return packages_found

    for file_info in requirement_files:

        if not isinstance(file_info, dict):
            continue

        content = file_info.get(
            "content",
            ""
        )

        if not isinstance(content, str):
            continue

        for line in content.splitlines():

            line = line.strip()

            if (
                not line
                or line.startswith("#")
                or line.startswith("-")
            ):
                continue

            match = re.match(
                r"^([A-Za-z0-9_.-]+)",
                line
            )

            if match:

                packages_found.add(
                    normalize(match.group(1))
                )

    return packages_found


# =========================================================
# PYPROJECT.TOML
# =========================================================

def extract_pyproject_packages(
    pyproject_files: Any
) -> set:

    packages_found = set()

    if not isinstance(pyproject_files, list):
        return packages_found

    for file_info in pyproject_files:

        if not isinstance(file_info, dict):
            continue

        content = file_info.get(
            "content",
            ""
        )

        if not isinstance(content, str):
            continue

        # Basic dependency extraction.
        # This intentionally avoids requiring another TOML package.

        dependency_section = False

        for line in content.splitlines():

            stripped = line.strip()

            if stripped.startswith("["):

                dependency_section = (
                    "dependencies" in stripped.lower()
                )

                continue

            if not dependency_section:
                continue

            match = re.match(
                r'^["\']?([A-Za-z0-9_.-]+)',
                stripped
            )

            if match:

                package_name = match.group(1)

                if package_name not in {
                    "dependencies",
                    "dev-dependencies",
                }:

                    packages_found.add(
                        normalize(package_name)
                    )

    return packages_found


# =========================================================
# DEPENDENCIES
# =========================================================

def extract_dependencies(
    dependencies: Dict[str, Any]
) -> List[Tuple[str, str]]:

    results = []

    if not isinstance(dependencies, dict):
        return results

    npm_packages = extract_npm_packages(
        dependencies.get("packageJson", [])
    )

    python_packages = extract_python_requirements(
        dependencies.get("requirementsTxt", [])
    )

    python_packages.update(
        extract_pyproject_packages(
            dependencies.get("pyprojectToml", [])
        )
    )

    # ---------------------------------------------
    # Match dependencies against taxonomy
    # ---------------------------------------------

    for skill_name, skill_data in SKILL_TAXONOMY.items():

        npm_names = {
            normalize(package)
            for package in skill_data.get(
                "npm",
                []
            )
        }

        pypi_names = {
            normalize(package)
            for package in skill_data.get(
                "pypi",
                []
            )
        }

        npm_matches = (
            npm_packages.intersection(
                npm_names
            )
        )

        python_matches = (
            python_packages.intersection(
                pypi_names
            )
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


# =========================================================
# COMPLETE STRUCTURED EXTRACTION
# =========================================================

def extract_structured_skills(
    repository: Dict[str, Any]
) -> List[Tuple[str, str]]:

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

    results.extend(
        extract_languages(languages)
    )

    results.extend(
        extract_topics(topics)
    )

    results.extend(
        extract_dependencies(dependencies)
    )

    return results