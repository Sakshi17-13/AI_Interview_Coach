"""Session-local orchestration for the adaptive interview AI components."""

from __future__ import annotations

from uuid import uuid4

from .answer_assessor import AnswerAssessor
from .interview_planner import InterviewPlanner
from .question_generator import QuestionGenerator
from .schemas import (
    AnswerAssessment,
    AnswerSubmissionResult,
    CandidateProfile,
    Competency,
    Evidence,
    InterviewAnswerRecord,
    InterviewPlanState,
    InterviewQuestion,
    InterviewSessionState,
    PreviousAnswerSignal,
    QuestionPlan,
    RoleProfile,
    SkillMatch,
    SkillMatchReport,
)


class InterviewEngineError(RuntimeError):
    """Raised for invalid interview-session operations or component failures."""


class InterviewEngine:
    """Coordinates planning, question wording, and answer assessment for one session.

    The component instances are injected for offline testing. No state is kept at module scope.
    """

    def __init__(
        self,
        planner: InterviewPlanner | None = None,
        question_generator: QuestionGenerator | None = None,
        answer_assessor: AnswerAssessor | None = None,
    ) -> None:
        self._planner = planner or InterviewPlanner()
        self._question_generator = question_generator or QuestionGenerator()
        self._answer_assessor = answer_assessor or AnswerAssessor()
        self._state: InterviewSessionState | None = None

    def initialize_interview(
        self,
        candidate_profile: CandidateProfile,
        role_profile: RoleProfile,
        skill_match_report: SkillMatchReport,
        target_question_count: int,
        session_id: str | None = None,
        max_follow_ups: int = 2,
    ) -> InterviewQuestion:
        """Create an isolated session and render the planner's first (introduction) question."""

        plan_state = self._planner.create_initial_state(
            role_profile, target_question_count, max_follow_ups
        )
        self._state = InterviewSessionState(
            session_id=session_id or str(uuid4()),
            candidate_profile=candidate_profile,
            role_profile=role_profile,
            skill_match_report=skill_match_report,
            plan_state=plan_state,
            remaining_questions=target_question_count,
        )
        try:
            question = self.get_next_question()
        except Exception:
            # Initialization can be retried with the same isolated profiles after a component failure.
            raise
        if question is None:
            raise InterviewEngineError("Interview initialization did not produce a first question.")
        return question

    def get_current_question(self) -> InterviewQuestion | None:
        return self._require_state().current_question

    def get_next_question(self) -> InterviewQuestion | None:
        """Return the current rendered question or safely render a planner-selected next question."""

        state = self._require_state()
        if state.status == "completed":
            return None
        if state.current_question is not None:
            return state.current_question

        try:
            plan, planned_state = self._planner.plan_next(
                state.candidate_profile,
                state.role_profile,
                state.skill_match_report,
                state.plan_state,
                state.pending_answer_signal,
            )
        except Exception as error:
            raise InterviewEngineError("Interview planning failed.") from error
        if plan is None:
            self._state = state.model_copy(
                update={"status": "completed", "current_question": None, "pending_answer_signal": None}
            )
            return None

        try:
            question = self._render_question(state, plan)
        except Exception as error:
            # Do not commit planned_state: the engine remains recoverable and can retry rendering.
            raise InterviewEngineError("Question generation failed.") from error

        self._state = state.model_copy(
            update={
                "plan_state": planned_state,
                "current_question": question,
                "generated_questions": [*state.generated_questions, question],
                "pending_answer_signal": None,
                "remaining_questions": planned_state.target_question_count
                - planned_state.questions_generated,
            }
        )
        return question

    def submit_answer(self, transcript: str | None) -> AnswerSubmissionResult:
        """Assess the current answer, update coverage, then render the next planned question."""

        state = self._require_state()
        if state.status == "completed":
            raise InterviewEngineError("Cannot submit an answer after interview completion.")
        current_question = state.current_question
        if current_question is None:
            raise InterviewEngineError("No current question is available for answer submission.")
        current_plan = self._find_plan(state.plan_state, current_question.question_id)
        competency = self._resolve_competency(state.role_profile, current_question.competency_id)
        match = self._resolve_match(state.skill_match_report, current_question.competency_id)
        job_evidence = competency.evidence if competency is not None else []
        candidate_evidence = match.candidate_evidence if match is not None else []
        exact_transcript = "" if transcript is None else transcript

        try:
            assessment = self._answer_assessor.assess(
                current_question,
                exact_transcript,
                competency,
                job_evidence,
                candidate_evidence,
                current_plan,
            )
        except Exception as error:
            raise InterviewEngineError("Answer assessment failed.") from error

        signal = self._signal_from_assessment(assessment)
        try:
            updated_plan_state = state.plan_state
            if signal is not None:
                updated_plan_state = self._planner.record_coverage(
                    state.plan_state,
                    signal,
                    was_follow_up=current_question.question_type == "follow_up",
                )
        except Exception as error:
            raise InterviewEngineError("Interview coverage update failed.") from error

        assessed_state = state.model_copy(
            update={
                "plan_state": updated_plan_state,
                "current_question": None,
                "answer_history": [
                    *state.answer_history,
                    InterviewAnswerRecord(
                        question_id=current_question.question_id,
                        transcript=exact_transcript,
                        assessment=assessment,
                    ),
                ],
                "pending_answer_signal": signal,
            }
        )
        self._state = assessed_state

        if updated_plan_state.questions_generated >= updated_plan_state.target_question_count:
            self._state = assessed_state.model_copy(
                update={"status": "completed", "pending_answer_signal": None}
            )
            return AnswerSubmissionResult(assessment=assessment, next_question=None, state=self._state)

        try:
            next_question = self.get_next_question()
        except InterviewEngineError:
            # Assessment and coverage have already been safely committed; caller can retry get_next_question.
            raise
        return AnswerSubmissionResult(assessment=assessment, next_question=next_question, state=self._require_state())

    def is_complete(self) -> bool:
        return self._require_state().status == "completed"

    def get_state(self) -> InterviewSessionState:
        return self._require_state().model_copy(deep=True)

    def get_assessments(self) -> list[AnswerAssessment]:
        return [record.assessment for record in self._require_state().answer_history]

    def _render_question(self, state: InterviewSessionState, plan: QuestionPlan) -> InterviewQuestion:
        competency = self._resolve_competency(state.role_profile, plan.competency_id)
        match = self._resolve_match(state.skill_match_report, plan.competency_id)
        previous_question = self._previous_question_text(state, plan.follow_up_of)
        return self._question_generator.generate(
            plan,
            competency,
            competency.evidence if competency is not None else [],
            match.candidate_evidence if match is not None else [],
            previous_question,
            state.pending_answer_signal if plan.question_type == "follow_up" else None,
        )

    @staticmethod
    def _find_plan(plan_state: InterviewPlanState, question_id: str) -> QuestionPlan:
        for plan in plan_state.question_history:
            if plan.question_id == question_id:
                return plan
        raise InterviewEngineError("Current question is missing its QuestionPlan.")

    @staticmethod
    def _resolve_competency(role: RoleProfile, competency_id: str | None) -> Competency | None:
        if competency_id is None:
            return None
        for competency in role.required_competencies + role.preferred_competencies:
            if competency.id == competency_id:
                return competency
        raise InterviewEngineError("Question references a competency absent from the role profile.")

    @staticmethod
    def _resolve_match(report: SkillMatchReport, competency_id: str | None) -> SkillMatch | None:
        if competency_id is None:
            return None
        for match in report.required_competency_results + report.preferred_competency_results:
            if match.competency_id == competency_id:
                return match
        raise InterviewEngineError("Question references a competency absent from the skill match report.")

    @staticmethod
    def _previous_question_text(state: InterviewSessionState, question_id: str | None) -> str | None:
        if question_id is None:
            return None
        for question in state.generated_questions:
            if question.question_id == question_id:
                return question.question_text
        raise InterviewEngineError("Follow-up question references a missing generated question.")

    @staticmethod
    def _signal_from_assessment(assessment: AnswerAssessment) -> PreviousAnswerSignal | None:
        if assessment.competency_id is None:
            return None
        return PreviousAnswerSignal(
            question_id=assessment.question_id,
            competency_id=assessment.competency_id,
            score=assessment.overall_score,
            gaps=assessment.gaps,
        )

    def _require_state(self) -> InterviewSessionState:
        if self._state is None:
            raise InterviewEngineError("Interview has not been initialized.")
        return self._state
