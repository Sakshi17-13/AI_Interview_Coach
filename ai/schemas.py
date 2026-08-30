"""Validated contracts for information extracted from application inputs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SchemaModel(BaseModel):
    """Base model that rejects fields the application has not defined."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Skill(SchemaModel):
    name: str = Field(min_length=1, description="Skill explicitly named in the resume.")
    proficiency_evidence: str | None = Field(
        default=None,
        description="Brief supporting evidence from the resume, if present.",
    )


class Experience(SchemaModel):
    id: str = Field(min_length=1, description="Stable identifier such as exp_1.")
    organization: str | None = None
    role: str | None = None
    duration: str | None = Field(default=None, description="Use only stated dates/duration.")
    highlights: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)


class Education(SchemaModel):
    degree: str | None = None
    field: str | None = None
    institution: str | None = None
    graduation_date: str | None = None


class CandidateProfile(SchemaModel):
    candidate_name: str | None = None
    headline: str | None = Field(
        default=None, description="Resume-supported professional summary or current title."
    )
    years_experience_estimate: float | None = Field(default=None, ge=0)
    skills: list[Skill] = Field(default_factory=list)
    experiences: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)


class Competency(SchemaModel):
    id: str = Field(min_length=1, description="Stable identifier such as python_data_work.")
    name: str = Field(min_length=1)
    requirements: list[str] = Field(default_factory=list)
    weight: float = Field(default=1.0, gt=0, le=1.0)


class RoleProfile(SchemaModel):
    job_title: str | None = None
    seniority: str | None = None
    location: str | None = None
    employment_type: str | None = None
    required_competencies: list[Competency] = Field(default_factory=list)
    preferred_competencies: list[Competency] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    interview_focus: list[str] = Field(default_factory=list)


class SkillMatch(SchemaModel):
    """Future-ready comparison of a candidate skill to a role competency."""

    competency_id: str = Field(min_length=1)
    competency_name: str = Field(min_length=1)
    match_level: Literal["strong", "partial", "not_evidenced"]
    supporting_evidence: list[str] = Field(default_factory=list)
    gap: str | None = None
