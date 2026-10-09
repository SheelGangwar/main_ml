import csv
import os
import re

INPUT_FILE = "data/issue_features.csv"
OUTPUT_FILE = "data/issues_with_technologies.csv"


TECHNOLOGIES = {
    "python": "Python",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "java": "Java",
    "c++": "C++",
    "cpp": "C++",
    "c#": "C#",
    "c": "C",
    "go": "Go",
    "golang": "Go",
    "rust": "Rust",
    "php": "PHP",
    "ruby": "Ruby",
    "kotlin": "Kotlin",
    "swift": "Swift",
    "scala": "Scala",
    "dart": "Dart",

    "react": "React",
    "reactjs": "React",
    "angular": "Angular",
    "vue": "Vue",
    "vue.js": "Vue",
    "next.js": "Next.js",
    "nextjs": "Next.js",
    "node.js": "Node.js",
    "nodejs": "Node.js",
    "express": "Express",
    "express.js": "Express",

    "django": "Django",
    "flask": "Flask",
    "fastapi": "FastAPI",
    "spring": "Spring",
    "laravel": "Laravel",
    "rails": "Ruby on Rails",

    "html": "HTML",
    "css": "CSS",
    "sass": "Sass",
    "tailwind": "Tailwind CSS",
    "bootstrap": "Bootstrap",

    "sql": "SQL",
    "mysql": "MySQL",
    "postgresql": "PostgreSQL",
    "postgres": "PostgreSQL",
    "mongodb": "MongoDB",
    "redis": "Redis",
    "sqlite": "SQLite",
    "oracle": "Oracle",

    "docker": "Docker",
    "kubernetes": "Kubernetes",
    "k8s": "Kubernetes",
    "terraform": "Terraform",
    "jenkins": "Jenkins",
    "github actions": "GitHub Actions",

    "aws": "AWS",
    "azure": "Azure",
    "gcp": "Google Cloud",
    "google cloud": "Google Cloud",

    "tensorflow": "TensorFlow",
    "pytorch": "PyTorch",
    "keras": "Keras",
    "scikit-learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "pandas": "Pandas",
    "numpy": "NumPy",

    "git": "Git",
    "github": "GitHub",
    "linux": "Linux"
}


def normalize(value):
    if not value:
        return ""

    return str(value).lower()


def extract_from_text(text):
    found = set()

    text = normalize(text)

    for keyword, technology in TECHNOLOGIES.items():

        if keyword in {"c", "c++", "c#"}:
            pattern = rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])"
        else:
            pattern = rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])"

        if re.search(pattern, text):
            found.add(technology)

    return found


def main():

    if not os.path.exists(INPUT_FILE):
        print(f"Input file not found: {INPUT_FILE}")
        return

    processed_rows = []

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            text = " ".join([
                row.get("combined_text", ""),
                row.get("labels", ""),
                row.get("repository_language", ""),
                row.get("repository_topics", "")
            ])

            technologies = extract_from_text(text)

            # Add repository language as a technology signal
            language = row.get("repository_language", "").strip()

            if language:
                language_lower = language.lower()

                if language_lower in TECHNOLOGIES:
                    technologies.add(
                        TECHNOLOGIES[language_lower]
                    )
                else:
                    technologies.add(language)

            row["technologies"] = "|".join(
                sorted(technologies)
            )

            row["technology_count"] = len(technologies)

            processed_rows.append(row)

    fieldnames = list(processed_rows[0].keys())

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
        writer.writerows(processed_rows)

    print("===================================")
    print("Technology extraction completed!")
    print(f"Processed issues: {len(processed_rows)}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("===================================")


if __name__ == "__main__":
    main()