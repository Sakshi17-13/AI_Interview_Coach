"""Build validated candidate and role profiles from raw application inputs."""

from __future__ import annotations

import re

from pydantic import BaseModel

from .llm_client import get_structured_client
from .prompts import job_description_analysis_messages, resume_analysis_messages
from .schemas import CandidateProfile, Evidence, RoleProfile


class ProfileGroundingError(ValueError):
    """Raised when returned evidence does not occur in the source document."""


def _normalise_for_evidence_check(text: str) -> str:
    """Make harmless whitespace and punctuation extraction differences comparable."""

    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def _iter_evidence(value: object):
    if isinstance(value, Evidence):
        yield value
    elif isinstance(value, BaseModel):
        for field_value in value.__dict__.values():
            yield from _iter_evidence(field_value)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_evidence(item)


def _validate_evidence_against_source(
    profile: CandidateProfile | RoleProfile, source_text: str, source_name: str
) -> None:
    normalised_source = _normalise_for_evidence_check(source_text)
    for evidence in _iter_evidence(profile):
        if _normalise_for_evidence_check(evidence.excerpt) not in normalised_source:
            raise ProfileGroundingError(
                f"Profile contains evidence not found in the {source_name}: "
                f"{evidence.excerpt!r}"
            )


def build_candidate_profile(resume_text: str) -> CandidateProfile:
    """Extract an evidence-grounded, resume-supported candidate profile."""

    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text is required to build a candidate profile.")
    profile = get_structured_client().generate_structured(
        resume_analysis_messages(resume_text, CandidateProfile.model_json_schema()),
        CandidateProfile,
    )
    _validate_evidence_against_source(profile, resume_text, "resume")
    return profile


def build_role_profile(job_description: str) -> RoleProfile:
    """Extract an evidence-grounded role profile for later matching."""

    if not job_description or not job_description.strip():
        raise ValueError("Job description is required to build a role profile.")
    profile = get_structured_client().generate_structured(
        job_description_analysis_messages(job_description, RoleProfile.model_json_schema()),
        RoleProfile,
    )
    _validate_evidence_against_source(profile, job_description, "job description")
    return profile
