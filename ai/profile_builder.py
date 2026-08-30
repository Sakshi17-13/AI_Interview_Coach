"""Build validated candidate and role profiles from raw application inputs."""

from __future__ import annotations

from .llm_client import get_structured_client
from .prompts import job_description_analysis_messages, resume_analysis_messages
from .schemas import CandidateProfile, RoleProfile


def build_candidate_profile(resume_text: str) -> CandidateProfile:
    """Extract resume-supported candidate information into a validated profile."""

    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text is required to build a candidate profile.")
    return get_structured_client().generate_structured(
        resume_analysis_messages(resume_text, CandidateProfile.model_json_schema()),
        CandidateProfile,
    )


def build_role_profile(job_description: str) -> RoleProfile:
    """Extract required/preferred role requirements into a validated profile."""

    if not job_description or not job_description.strip():
        raise ValueError("Job description is required to build a role profile.")
    return get_structured_client().generate_structured(
        job_description_analysis_messages(job_description, RoleProfile.model_json_schema()),
        RoleProfile,
    )
