
from recommendation.embedding_service import cosine_similarity


def normalize(value: str) -> str:
    return " ".join((value or "").lower().replace("-", " ").split())


def unique_names(values):
    result = []
    seen = set()

    for value in values or []:
        name = str(value).strip()
        key = normalize(name)

        if key and key not in seen:
            seen.add(key)
            result.append(name)

    return result


def recommend_issues(student, issues):
    student_skills = student.skills or []

    skill_names = [
        skill.name.strip()
        for skill in student_skills
        if skill.name and skill.name.strip()
    ]

    skill_lookup = {
        normalize(skill.name): max(0.0, min(1.0, skill.confidence))
        for skill in student_skills
        if skill.name
    }

    student_text_parts = []

    for skill in student_skills:
        evidence = ", ".join(skill.evidence or [])
        student_text_parts.append(
            f"{skill.name}. Evidence: {evidence}"
        )

    student_text = " ".join(student_text_parts)

    results = []

    for issue in issues:
        technologies = unique_names(issue.get("technologies", []))
        concepts = unique_names(issue.get("concepts", []))
        
        print("Issue:", issue.get("title"))
        print("Technologies:", technologies)
        print("Concepts:", concepts)

        issue_text = " ".join([
            issue.get("title", ""),
            issue.get("description", ""),
            "Technologies: " + ", ".join(technologies),
            "Concepts: " + ", ".join(concepts),
            "Difficulty: " + issue.get("difficulty", "unknown"),
        ]).strip()

        matched_skills = []
        missing_skills = []
        overlap_scores = []

        for technology in technologies:
            key = normalize(technology)

            if key in skill_lookup:
                matched_skills.append(technology)
                overlap_scores.append(skill_lookup[key])
            else:
                missing_skills.append(technology)

        # Skill overlap: higher when the student has evidence
        # for more of the issue's required technologies.
        if technologies:
            overlap_score = len(matched_skills) / len(technologies)
        else:
            overlap_score = 0.0

        # Confidence-weighted overlap complements the simple match count.
        confidence_score = (
            sum(overlap_scores) / len(technologies)
            if technologies else 0.0
        )

        semantic_score = cosine_similarity(
            student_text,
            issue_text
        ) if student_text and issue_text else 0.0

        # Concepts matching a known student skill add extra evidence.
        known_names = set(skill_lookup)
        concept_matches = [
            concept for concept in concepts
            if normalize(concept) in known_names
        ]

        concept_score = (
            len(concept_matches) / len(concepts)
            if concepts else 0.0
        )

        # Weighted ranking score, normalized to the 0–1 range.
        score = (
            

           0.25 * semantic_score
           + 0.40 * overlap_score
           + 0.20 * confidence_score
           + 0.15 * concept_score

        )

        reasons = []

        if matched_skills:
            reasons.append(
                "Matches your skills: "
                + ", ".join(matched_skills[:4])
            )

        if concept_matches:
            reasons.append(
                "Related concepts: "
                + ", ".join(concept_matches[:3])
            )

        if semantic_score >= 0.60:
            reasons.append("Strong semantic match with your profile")
        elif semantic_score >= 0.35:
            reasons.append("Some semantic similarity with your profile")

        if not reasons:
            reasons.append(
                "Ranked using profile similarity and available issue features"
            )

        results.append({
            "issueId": issue["id"],
            "score": round(max(0.0, min(1.0, score)), 4),
            "matchedSkills": matched_skills,
            "missingSkills": missing_skills,
            "reason": "; ".join(reasons),
        })

    results.sort(key=lambda item: item["score"], reverse=True)
    return results