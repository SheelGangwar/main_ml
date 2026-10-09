from fastapi import (
    FastAPI,
    HTTPException,
    status
)

from schemas import (
    AnalyzeProfileRequest,
    AnalyzeProfileResponse,
    RepoData,
    SkillGapRequest,
    SkillGapResponse,
    GapDetail,
    AnalyzeIssueRequest,
    AnalyzeIssueResponse,
    IssueAnalysisDetail,
    RecommendRequest,
    RecommendResponse,
    RecommendationItem)

from services.profile_analyzer import (
    analyze_repositories,
    analyze_repository
)

# NEW ML Issue Analyz
from services.issue_analyzer import (
    analyze_issue as analyze_issue_ml
)

from services.recommendation_engine import (
    recommend_issues as rank_issues
)

app = FastAPI(
    title="OpenSourceMentor ML Service",
    version="1.0.0",
    description="Internal ML microservice for profile analysis and issue matching"
)


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health_check():

    return {
        "status": "ok",
        "service": "ML Engine"
    }


# =========================================================
# SINGLE REPOSITORY ANALYSIS
# =========================================================

@app.post(
    "/ml/analyze-repository"
)
def analyze_single_repository(
    payload: RepoData
):

    try:

        skills = analyze_repository(
            payload
        )

        return {
            "success": True,
            "skills": skills
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Repository analysis failed: {str(e)}"
        )


# =========================================================
# PROFILE ANALYSIS
# =========================================================

@app.post(
    "/ml/analyze-profile",
    response_model=AnalyzeProfileResponse
)
def analyze_profile(
    payload: AnalyzeProfileRequest
):

    try:

        skills = analyze_repositories(
            payload.repositories
        )

        return AnalyzeProfileResponse(
            success=True,
            skills=skills
        )

    except Exception as e:

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Profile analysis failed: {str(e)}"
        )


# =========================================================
# SKILL GAP
# =========================================================

@app.post(
    "/ml/skill-gap",
    response_model=SkillGapResponse
)
def skill_gap(
    payload: SkillGapRequest
):

    existing_skills = {
        s.name.lower()
        for s in payload.studentSkills
    }

    gaps = []

    for target in payload.targetSkills:

        target_lower = target.lower()

        if target_lower not in existing_skills:

            priority = (
                "high"
                if target_lower in [
                    "typescript",
                    "testing"
                ]
                else "medium"
            )

            gaps.append(
                GapDetail(
                    skill=target,
                    priority=priority
                )
            )

    return SkillGapResponse(
        success=True,
        gaps=gaps
    )


# =========================================================
# ISSUE ANALYSIS - NEW ML PIPELINE
# =========================================================

@app.post(
    "/ml/analyze-issue",
    response_model=AnalyzeIssueResponse
)
def analyze_issue(
    payload: AnalyzeIssueRequest
):

    try:

        result = analyze_issue_ml(
            title=payload.title,
            description=payload.description,
            labels=", ".join(payload.labels),
            repository_language=(
                payload.repository.language
                if payload.repository
                else ""
            )
        )

        return AnalyzeIssueResponse(
            success=True,
            analysis=IssueAnalysisDetail(
                technologies=result["technologies"],
                difficulty=result["difficulty"],
                concepts=result["concepts"]
            )
        )

    except Exception as e:

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Issue analysis failed: {str(e)}"
        )


# =========================================================
# RECOMMENDATIONS
# =========================================================

# =========================================================
# RECOMMENDATIONS
# =========================================================

@app.post(
    "/ml/recommend",
    response_model=RecommendResponse
)
def recommend_issues(
    payload: RecommendRequest
):

    try:
        analyzed_issues = []

        # Step 1: Analyze each issue using the trained ML pipeline.
        for issue in payload.issues:

            result = analyze_issue_ml(
                title=issue.title or "",
                description=issue.description or "",
                labels=", ".join(issue.labels),
                repository_language=issue.repository_language or ""
            )

            technologies = list(dict.fromkeys(
                (issue.technologies or [])
                + result.get("technologies", [])
            ))

            analyzed_issues.append({
                "id": issue.id,
                "title": issue.title or "",
                "description": issue.description or "",
                "technologies": technologies,
                "concepts": result.get("concepts", []),
                "difficulty": result.get(
                    "difficulty",
                    issue.difficulty or "unknown"
                )
            })

        # Step 2: Rank issues after analysis is complete.
        recommendations = rank_issues(
            student=payload.student,
            issues=analyzed_issues
        )

        # Step 3: Return the existing response schema.
        return RecommendResponse(
            success=True,
            recommendations=[
                RecommendationItem(**item)
                for item in recommendations
            ]
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Recommendation failed: {str(e)}"
        )
# =========================================================
# LOCAL DEVELOPMENT
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )