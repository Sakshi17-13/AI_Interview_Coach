"""Thin, session-local application adapter around the completed AI backend.

This module deliberately contains no interview decision logic.  It gives the
Gradio layer a small, testable object that coordinates the public AI APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ai import FinalReportBuilder, InterviewEngine, SkillMatcher
from ai.schemas import AnswerAssessment, CandidateProfile, FinalInterviewReport, InterviewQuestion, RoleProfile, SkillMatchReport


@dataclass
class InterviewTurn:
    """UI-safe outcome of one submitted transcript."""

    assessment: AnswerAssessment
    next_question: InterviewQuestion | None
    report: FinalInterviewReport | None


class InterviewApplicationSession:
    """One independently usable application interview session."""

    def __init__(
        self,
        engine: Any,
        report_builder: Any,
        candidate_profile: CandidateProfile,
        role_profile: RoleProfile,
        skill_match_report: SkillMatchReport,
    ) -> None:
        self.engine = engine
        self.report_builder = report_builder
        self.candidate_profile = candidate_profile
        self.role_profile = role_profile
        self.skill_match_report = skill_match_report
        self._needs_question_retry = False
        self._ended_early = False

    def submit_transcript(self, transcript: str) -> InterviewTurn:
        """Pass the transcript unchanged to the engine and report only when complete."""

        if self._ended_early:
            raise RuntimeError("This interview was ended early and cannot accept more answers.")
        try:
            result = self.engine.submit_answer(transcript)
        except Exception:
            # InterviewEngine commits a completed assessment before a later question-rendering
            # failure. Mark only that recoverable case for a question-only retry.
            if not self.engine.is_complete() and self.engine.get_current_question() is None:
                self._needs_question_retry = True
            raise
        report = self.report_builder.build_from_engine(self.engine) if self.engine.is_complete() else None
        return InterviewTurn(result.assessment, result.next_question, report)

    @property
    def needs_question_retry(self) -> bool:
        return self._needs_question_retry

    @property
    def ended_early(self) -> bool:
        return self._ended_early

    def retry_next_question(self) -> InterviewQuestion | None:
        """Render a pending next question without repeating the previous assessment."""

        if not self._needs_question_retry:
            return self.engine.get_current_question()
        question = self.engine.get_next_question()
        self._needs_question_retry = question is None and not self.engine.is_complete()
        return question

    def build_final_report(self) -> FinalInterviewReport:
        """Retry report creation from the engine's unchanged state."""

        return self.report_builder.build_from_engine(self.engine)

    def end_early(self) -> FinalInterviewReport:
        """Stop UI submissions and return the backend's explicitly incomplete report."""

        self._ended_early = True
        return self.build_final_report()


def create_interview_session(
    candidate_profile: CandidateProfile,
    role_profile: RoleProfile,
    target_question_count: int,
    *,
    skill_matcher: Any | None = None,
    engine_factory: Callable[[], Any] = InterviewEngine,
    report_builder: Any | None = None,
    session_id: str | None = None,
) -> tuple[InterviewApplicationSession, InterviewQuestion]:
    """Create one engine, match profiles, and obtain its first planned question."""

    matcher = skill_matcher or SkillMatcher()
    match_report = matcher.match(candidate_profile, role_profile)
    engine = engine_factory()
    first_question = engine.initialize_interview(
        candidate_profile,
        role_profile,
        match_report,
        int(target_question_count),
        session_id=session_id,
    )
    return (
        InterviewApplicationSession(
            engine,
            report_builder or FinalReportBuilder(),
            candidate_profile,
            role_profile,
            match_report,
        ),
        first_question,
    )
