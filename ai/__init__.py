"""Structured AI foundation for the interview coach.

The package deliberately has no Gradio or audio dependencies.  The existing
application supplies its already-initialised watsonx.ai model through
``configure_structured_client``.
"""

from .llm_client import configure_structured_client
from .profile_builder import build_candidate_profile, build_role_profile
from .schemas import CandidateProfile, RoleProfile

__all__ = [
    "CandidateProfile",
    "RoleProfile",
    "build_candidate_profile",
    "build_role_profile",
    "configure_structured_client",
]
