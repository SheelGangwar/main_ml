print("TEST FILE STARTED")


from services.skill_extraction.structured_extractor import (
    extract_structured_skills
)

from services.skill_extraction.semantic_extractor import (
    extract_semantic_skills
)

from services.skill_extraction.evidence_aggregator import (
    aggregate_skill_evidence
)


print("IMPORTS DONE")


# --------------------------------
# Test repository
# --------------------------------
repository = {
    "name": "full-stack-test",

    "languages": {
        "JavaScript": 60000,
        "HTML": 20000,
        "CSS": 10000
    },

    "topics": [
        "react",
        "nodejs",
        "docker"
    ],

    "dependencies": {
    "packageJson": {
        "dependencies": {
            "react": "^18.3.0",
            "express": "^4.19.0",
            "mongoose": "^8.5.0"
        }
    },

    "requirementsTxt": {},

    "pyprojectToml": {}
}
}


# --------------------------------
# Structured extraction
# --------------------------------

structured_results = extract_structured_skills(
    repository
)

print("\nSTRUCTURED RESULTS")

for skill, evidence in structured_results:

    print(
        f"{skill} -> {evidence}"
    )


# --------------------------------
# Semantic extraction
# --------------------------------

readme = """
A full-stack web application built with React,
Node.js and Express. MongoDB is used as the database.
The application is containerized using Docker.
"""


semantic_skills = extract_semantic_skills(
    readme
)

print("\nSEMANTIC RESULTS")

for skill, score, evidence in semantic_skills:

    print(
        f"{skill} -> {score:.3f} -> {evidence}"
    )


# --------------------------------
# Evidence aggregation
# --------------------------------

final_results = aggregate_skill_evidence(
    structured_results,
    semantic_skills
)

print("\nFINAL AGGREGATED RESULTS")

for result in final_results:

    print(result)