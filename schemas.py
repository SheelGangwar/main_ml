from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field


# =========================================================
# PROFILE / REPOSITORY ANALYSIS
# =========================================================

class PackageJsonFile(BaseModel):
    path: str = ""
    data: Optional[Dict[str, Any]] = None


class RequirementsFile(BaseModel):
    path: str = ""
    content: Optional[str] = ""


class PyprojectFile(BaseModel):
    path: str = ""
    content: Optional[str] = ""


class DependencyInfo(BaseModel):
    packageJson: List[PackageJsonFile] = Field(
        default_factory=list
    )

    requirementsTxt: List[RequirementsFile] = Field(
        default_factory=list
    )

    pyprojectToml: List[PyprojectFile] = Field(
        default_factory=list
    )


class RepoData(BaseModel):
    name: str

    languages: Dict[str, int] = Field(
        default_factory=dict
    )

    topics: List[str] = Field(
        default_factory=list
    )

    readme: Optional[str] = ""

    dependencies: DependencyInfo = Field(
        default_factory=DependencyInfo
    )


class AnalyzeProfileRequest(BaseModel):
    userId: str

    repositories: List[RepoData]


class SkillConfidence(BaseModel):
    name: str

    confidence: float

    evidence: List[str] = Field(
        default_factory=list
    )

    source: Optional[str] = ""


class AnalyzeProfileResponse(BaseModel):
    success: bool = True

    skills: List[SkillConfidence]


# =========================================================
# SKILL GAP
# =========================================================

class SkillGapRequest(BaseModel):
    studentSkills: List[SkillConfidence]

    targetSkills: List[str]


class GapDetail(BaseModel):
    skill: str

    priority: str


class SkillGapResponse(BaseModel):
    success: bool = True

    gaps: List[GapDetail]


# =========================================================
# ISSUE ANALYSIS
# =========================================================

class RepositoryInfo(BaseModel):
    name: str

    language: Optional[str] = None


class AnalyzeIssueRequest(BaseModel):
    issueId: str

    title: str

    description: Optional[str] = ""

    labels: List[str] = Field(
        default_factory=list
    )

    repository: Optional[RepositoryInfo] = None


class IssueAnalysisDetail(BaseModel):
    technologies: List[str]

    difficulty: str

    concepts: List[str]


class AnalyzeIssueResponse(BaseModel):
    success: bool = True

    analysis: IssueAnalysisDetail


# =========================================================
# RECOMMENDATIONS
# =========================================================

class StudentPayload(BaseModel):
    skills: List[SkillConfidence]



class CandidateIssue(BaseModel):
    id: str

    technologies: List[str] = Field(
        default_factory=list
    )

    difficulty: str = "unknown"

    description: Optional[str] = ""

    title: Optional[str] = ""

    labels: List[str] = Field(
        default_factory=list
    )

    repository_language: Optional[str] = ""


class RecommendRequest(BaseModel):
    student: StudentPayload

    issues: List[CandidateIssue]


class RecommendationItem(BaseModel):
    issueId: str

    score: float

    matchedSkills: List[str]

    missingSkills: List[str]

    reason: str


class RecommendResponse(BaseModel):
    success: bool = True

    recommendations: List[RecommendationItem]