from typing import Dict, List, Any

from schemas import RepoData, SkillConfidence

from services.skill_extraction.structured_extractor import (
    extract_structured_skills
)

from services.skill_extraction.semantic_extractor import (
    extract_semantic_skills
)

from services.skill_extraction.evidence_aggregator import (
    aggregate_skill_evidence
)
from services.skill_extraction.skill_features import (
    build_student_skill_features
)


def repo_to_dict(repo: RepoData) -> Dict[str, Any]:
    """
    Convert Pydantic repository data into the dictionary
    expected by the skill extraction pipeline.
    """

    dependencies = {}

    if repo.dependencies:

        dependencies = {
            "packageJson": repo.dependencies.packageJson or {},
            "requirementsTxt": repo.dependencies.requirementsTxt or {},
            "pyprojectToml": repo.dependencies.pyprojectToml or {}
        }

    return {
        "name": repo.name,
        "languages": repo.languages or {},
        "topics": repo.topics or [],
        "readme": repo.readme or "",
        "dependencies": dependencies
    }


def analyze_repository(repo: RepoData) -> List[Dict]:
    """
    Analyze a single GitHub repository and extract its skills.
    """

    repository = repo_to_dict(repo)

    structured_results = extract_structured_skills(
        repository
    )

    semantic_results = extract_semantic_skills(
        repository.get("readme", "")
    )

    final_results = aggregate_skill_evidence(
        structured_results,
        semantic_results
    )

    return final_results


def analyze_repositories(
    repositories: List[RepoData]
) -> List[SkillConfidence]:

    student_skills: Dict[str, Dict] = {}

    # Store skill extraction results from every repository
    all_repo_results = []

    for repo in repositories:

        repo_results = analyze_repository(repo)

        # Keep repository-wise results for feature generation
        all_repo_results.append(repo_results)

        for result in repo_results:

            skill_name = result["name"]
            confidence = result["confidence"]
            evidence = result["evidence"]

            if skill_name not in student_skills:

                student_skills[skill_name] = {
                    "confidences": [],
                    "evidence": []
                }

            student_skills[skill_name]["confidences"].append(
                confidence
            )

            student_skills[skill_name]["evidence"].extend(
                evidence
            )
        # -----------------------------------------
    # Build ML-ready skill features
    # -----------------------------------------

    skill_features = build_student_skill_features(
        all_repo_results
    )

    print("\n===== STUDENT SKILL FEATURES =====")

    for feature in skill_features:
        print(feature)

    print("==================================\n")

    results = []

    for skill_name, data in student_skills.items():

        confidences = data["confidences"]

        confidence = max(confidences)

        results.append(
            SkillConfidence(
                name=skill_name,
                confidence=round(confidence, 3),
                evidence=data["evidence"]
            )
        )

    results.sort(
        key=lambda x: x.confidence,
        reverse=True
    )

    return results