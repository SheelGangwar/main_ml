from services.skill_extraction.skill_features import build_student_skill_features

all_repo_results = [
    [
        {
            "name": "React",
            "confidence": 0.56,
            "evidence": [
                "GitHub topic 'react' matches this skill",
                "Dependency 'react' found in package.json",
                "README semantic similarity with React: 0.441"
            ]
        },
        {
            "name": "JavaScript",
            "confidence": 0.45,
            "evidence": [
                "GitHub repository contains JavaScript code"
            ]
        }
    ],

    [
        {
            "name": "React",
            "confidence": 0.72,
            "evidence": [
                "Dependency 'react' found in package.json",
                "GitHub repository contains JavaScript code"
            ]
        }
    ]
]

features = build_student_skill_features(
    all_repo_results
)

for item in features:
    print(item)