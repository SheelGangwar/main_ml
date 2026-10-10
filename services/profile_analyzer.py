from typing import Dict, List, Any

from schemas import (
    RepoData,
    SkillConfidence
)

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


def repo_to_dict(
    repo: RepoData
) -> Dict[str, Any]:

    return {
        "name": repo.name,

        "languages": repo.languages or {},

        "topics": repo.topics or [],

        "readme": repo.readme or "",

        "dependencies": {
            "packageJson": [
                file.model_dump()
                for file in repo.dependencies.packageJson
            ],

            "requirementsTxt": [
                file.model_dump()
                for file in repo.dependencies.requirementsTxt
            ],

            "pyprojectToml": [
                file.model_dump()
                for file in repo.dependencies.pyprojectToml
            ],
        },
    }


def analyze_repository(
    repo: RepoData
) -> List[Dict]:

    repository = repo_to_dict(repo)

    structured_results = (
        extract_structured_skills(
            repository
        )
    )

    semantic_results = (
        extract_semantic_skills(
            repository.get(
                "readme",
                ""
            )
        )
    )

    final_results = (
        aggregate_skill_evidence(
            structured_results,
            semantic_results
        )
    )

    return final_results


def analyze_repositories(
    repositories: List[RepoData]
) -> List[SkillConfidence]:

    student_skills: Dict[
        str,
        Dict
    ] = {}

    all_repo_results = []

    for repo in repositories:

        repo_results = analyze_repository(
            repo
        )

        all_repo_results.append(
            repo_results
        )

        for result in repo_results:

            skill_name = result[
                "name"
            ]

            confidence = result[
                "confidence"
            ]

            evidence = result.get(
                "evidence",
                []
            )

            source = result.get(
                "source",
                "hybrid"
            )

            if skill_name not in student_skills:

                student_skills[skill_name] = {
                    "confidences": [],
                    "evidence": [],
                    "sources": set(),
                }

            student_skills[
                skill_name
            ][
                "confidences"
            ].append(
                confidence
            )

            student_skills[
                skill_name
            ][
                "evidence"
            ].extend(
                evidence
            )

            student_skills[
                skill_name
            ][
                "sources"
            ].add(
                source
            )

    # ==========================================
    # FEATURE GENERATION
    # ==========================================

    skill_features = (
        build_student_skill_features(
            all_repo_results
        )
    )

    print(
        "\n===== STUDENT SKILL FEATURES ====="
    )

    for feature in skill_features:

        print(feature)

    print(
        "==================================\n"
    )

    # ==========================================
    # FINAL STUDENT SKILLS
    # ==========================================

    results = []

    for skill_name, data in (
        student_skills.items()
    ):

        confidence = max(
            data["confidences"]
        )

        sources = data[
            "sources"
        ]

        source = (
            next(iter(sources))
            if len(sources) == 1
            else "hybrid"
        )

        results.append(
            SkillConfidence(
                name=skill_name,

                confidence=round(
                    confidence,
                    3
                ),

                evidence=list(
                    dict.fromkeys(
                        data["evidence"]
                    )
                ),

                source=source,
            )
        )

    results.sort(
        key=lambda x: x.confidence,
        reverse=True
    )

    return results