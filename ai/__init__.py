"""Structured AI foundation for the interview coach.

The package deliberately has no Gradio or audio dependencies.  The existing
application supplies its already-initialised watsonx.ai model through
``configure_structured_client``.
"""

from .llm_client import configure_structured_client
from .answer_assessor import AnswerAssessor
from .interview_engine import InterviewEngine
from .interview_planner import InterviewPlanner
from .profile_builder import build_candidate_profile, build_role_profile
from .question_generator import QuestionGenerator
from .report_builder import FinalReportBuilder
from .schemas import AnswerAssessment, CandidateProfile, FinalInterviewReport, InterviewPlanState, InterviewQuestion, InterviewSessionState, QuestionPlan, RoleProfile, SkillMatchReport
from .skill_matcher import SkillMatcher

__all__ = [
    "CandidateProfile",
    "RoleProfile",
    "AnswerAssessment",
    "AnswerAssessor",
    "InterviewEngine",
    "FinalReportBuilder",
    "FinalInterviewReport",
    "SkillMatchReport",
    "SkillMatcher",
    "InterviewPlanner",
    "InterviewPlanState",
    "QuestionPlan",
    "InterviewQuestion",
    "InterviewSessionState",
    "QuestionGenerator",
    "build_candidate_profile",
    "build_role_profile",
    "configure_structured_client",
]
