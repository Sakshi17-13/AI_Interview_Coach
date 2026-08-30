"""Validated contracts for information extracted from application inputs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SchemaModel(BaseModel):
    """Base model that rejects fields the application has not defined."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Evidence(SchemaModel):
    """A short excerpt that lets later stages trace a claim to its source input."""

    excerpt: str = Field(min_length=1, description="Verbatim or near-verbatim source excerpt.")


class Skill(SchemaModel):
    name: str = Field(min_length=1, description="Skill explicitly named in the resume.")
    evidence: list[Evidence] = Field(min_length=1)


class Technology(SchemaModel):
    """A named tool, platform, framework, library, or language from a resume."""

    name: str = Field(min_length=1)
    category: str | None = Field(
        default=None,
        description="Use only a clearly supported category such as language, cloud, or tool.",
    )
    evidence: list[Evidence] = Field(min_length=1)


class Achievement(SchemaModel):
    description: str = Field(min_length=1)
    context: str | None = None
    evidence: list[Evidence] = Field(min_length=1)


class Experience(SchemaModel):
    id: str = Field(min_length=1, description="Stable identifier such as exp_1.")
    organization: str | None = None
    role: str | None = None
    duration: str | None = Field(default=None, description="Use only stated dates/duration.")
    highlights: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    achievements: list[Achievement] = Field(default_factory=list)
    evidence: list[Evidence] = Field(min_length=1)


class Project(SchemaModel):
    id: str = Field(min_length=1, description="Stable identifier such as project_1.")
    name: str = Field(min_length=1)
    description: str | None = None
    technologies: list[str] = Field(default_factory=list)
    achievements: list[Achievement] = Field(default_factory=list)
    evidence: list[Evidence] = Field(min_length=1)


class Certification(SchemaModel):
    name: str = Field(min_length=1)
    issuer: str | None = None
    date: str | None = None
    credential_id: str | None = None
    evidence: list[Evidence] = Field(min_length=1)


class Education(SchemaModel):
    degree: str | None = None
    field: str | None = None
    institution: str | None = None
    graduation_date: str | None = None
    evidence: list[Evidence] = Field(min_length=1)


class CandidateProfile(SchemaModel):
    candidate_name: str | None = None
    candidate_name_evidence: list[Evidence] = Field(default_factory=list)
    headline: str | None = Field(
        default=None, description="Resume-supported professional summary or current title."
    )
    headline_evidence: list[Evidence] = Field(default_factory=list)
    years_experience_estimate: float | None = Field(default=None, ge=0)
    years_experience_evidence: list[Evidence] = Field(default_factory=list)
    skills: list[Skill] = Field(default_factory=list)
    technologies_tools: list[Technology] = Field(default_factory=list)
    experiences: list[Experience] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    achievements: list[Achievement] = Field(default_factory=list)
    certifications: list[Certification] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_evidence_for_profile_facts(self) -> "CandidateProfile":
        fact_evidence = (
            ("candidate_name", self.candidate_name, self.candidate_name_evidence),
            ("headline", self.headline, self.headline_evidence),
            (
                "years_experience_estimate",
                self.years_experience_estimate,
                self.years_experience_evidence,
            ),
        )
        for field_name, value, evidence in fact_evidence:
            if value is not None and not evidence:
                raise ValueError(f"{field_name} requires supporting resume evidence.")
        return self


class Competency(SchemaModel):
    id: str = Field(min_length=1, description="Stable identifier such as python_data_work.")
    name: str = Field(min_length=1)
    requirements: list[str] = Field(default_factory=list)
    weight: float = Field(
        default=1.0,
        gt=0,
        le=5.0,
        description="Relative importance, grounded only in explicit job-description priority signals.",
    )
    evidence: list[Evidence] = Field(min_length=1)


class TechnologyRequirement(SchemaModel):
    name: str = Field(min_length=1)
    requirement_level: Literal["required", "preferred"]
    evidence: list[Evidence] = Field(min_length=1)


class Responsibility(SchemaModel):
    description: str = Field(min_length=1)
    evidence: list[Evidence] = Field(min_length=1)


class InterviewFocus(SchemaModel):
    topic: str = Field(min_length=1)
    competency_ids: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(min_length=1)


class RoleProfile(SchemaModel):
    job_title: str | None = None
    seniority: str | None = None
    location: str | None = None
    employment_type: str | None = None
    required_competencies: list[Competency] = Field(default_factory=list)
    preferred_competencies: list[Competency] = Field(default_factory=list)
    technologies_tools: list[TechnologyRequirement] = Field(default_factory=list)
    responsibilities: list[Responsibility] = Field(default_factory=list)
    interview_focus: list[InterviewFocus] = Field(default_factory=list)

    @model_validator(mode="after")
    def keep_competency_levels_distinct(self) -> "RoleProfile":
        required_ids = {competency.id for competency in self.required_competencies}
        preferred_ids = {competency.id for competency in self.preferred_competencies}
        duplicates = required_ids.intersection(preferred_ids)
        if duplicates:
            raise ValueError(
                "A competency cannot be both required and preferred: "
                f"{', '.join(sorted(duplicates))}."
            )
        return self


class SkillMatch(SchemaModel):
    """Future-ready comparison of a candidate skill to a role competency."""

    competency_id: str = Field(min_length=1)
    competency_name: str = Field(min_length=1)
    match_level: Literal["strong", "partial", "not_evidenced"]
    supporting_evidence: list[str] = Field(default_factory=list)
    gap: str | None = None
