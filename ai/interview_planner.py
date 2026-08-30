"""Deterministic interview planning informed by profile matching and coverage state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .llm_client import get_structured_client
from .prompts import question_guidance_messages
from .schemas import (
    CandidateProfile,
    Competency,
    Evidence,
    InterviewCoverage,
    InterviewPlanState,
    PreviousAnswerSignal,
    QuestionGuidance,
    QuestionPlan,
    RoleProfile,
    SkillMatch,
    SkillMatchReport,
)


class InterviewPlanningError(ValueError):
    """Raised when profiles, reports, or planner state are inconsistent."""


RequirementLevel = Literal["required", "preferred"]


@dataclass(frozen=True)
class _RankedCompetency:
    competency: Competency
    match: SkillMatch
    level: RequirementLevel
    coverage: InterviewCoverage


class InterviewPlanner:
    """Ranks competencies in Python and asks the LLM only for bounded question guidance."""

    def create_initial_state(
        self, role_profile: RoleProfile, target_question_count: int, max_follow_ups: int = 2
    ) -> InterviewPlanState:
        competencies = role_profile.required_competencies + role_profile.preferred_competencies
        return InterviewPlanState(
            target_question_count=target_question_count,
            max_follow_ups=max_follow_ups,
            coverage=[InterviewCoverage(competency_id=competency.id) for competency in competencies],
        )

    def plan_next(
        self,
        candidate_profile: CandidateProfile,
        role_profile: RoleProfile,
        match_report: SkillMatchReport,
        state: InterviewPlanState,
        previous_answer_signal: PreviousAnswerSignal | None = None,
    ) -> tuple[QuestionPlan | None, InterviewPlanState]:
        """Plan one question or mark a completed interview as closed.

        Candidate-to-role matching and interview-response coverage are intentionally separate:
        ``match_report`` selects role relevance while ``state.coverage`` records only interview turns.
        """

        self._validate_inputs(role_profile, match_report, state, previous_answer_signal)
        if state.questions_generated >= state.target_question_count:
            return None, state.model_copy(update={"current_stage": "closing"})

        is_first_question = state.questions_generated == 0
        is_final_slot = (
            state.questions_generated == state.target_question_count - 1 and not is_first_question
        )
        if is_first_question:
            plan = self._static_plan(
                state, stage="introduction", question_type="open_ended", difficulty=1,
                objective="Introduce the candidate and establish interview context.",
                expected_evidence=["A concise professional introduction relevant to the target role."],
                reason="The first interview question is always an introduction.",
            )
        elif is_final_slot:
            plan = self._static_plan(
                state, stage="closing", question_type="open_ended", difficulty=1,
                objective="Give the candidate an opportunity to ask final role-related questions.",
                expected_evidence=[],
                reason="The final planned slot is reserved for closing.",
            )
        else:
            ranked = self._rank_competencies(role_profile, match_report, state)
            if not ranked:
                plan = QuestionPlan(
                    question_id=self._next_question_id(state),
                    competency_id=None,
                    stage="behavioral",
                    question_type="open_ended",
                    difficulty=2,
                    objective="Assess general communication and problem-solving approach.",
                    expected_evidence=["A clear example of how the candidate approached a professional challenge."],
                    follow_up_of=None,
                    reason="No role competencies are available, so use a safe general interview question.",
                )
                return plan, self._append_plan(state, plan)
            selected, follow_up_of = self._select_target(ranked, previous_answer_signal, state)
            stage, question_type, difficulty = self._select_format(
                candidate_profile, selected, state, follow_up_of is not None
            )
            guidance = self._generate_guidance(selected, previous_answer_signal)
            reason = self._selection_reason(selected, follow_up_of is not None)
            plan = QuestionPlan(
                question_id=self._next_question_id(state),
                competency_id=selected.competency.id,
                stage=stage,
                question_type=question_type,
                difficulty=difficulty,
                objective=guidance.objective,
                expected_evidence=guidance.expected_evidence,
                follow_up_of=follow_up_of,
                reason=reason,
            )
        return plan, self._append_plan(state, plan)

    def record_coverage(
        self, state: InterviewPlanState, signal: PreviousAnswerSignal
    ) -> InterviewPlanState:
        """Update interview-only coverage from a later assessment without changing resume matching."""

        coverage_items: list[InterviewCoverage] = []
        found = False
        for coverage in state.coverage:
            if coverage.competency_id != signal.competency_id:
                coverage_items.append(coverage)
                continue
            found = True
            status = "covered" if signal.score >= 4 else "partial"
            coverage_items.append(
                coverage.model_copy(
                    update={
                        "questions_asked": coverage.questions_asked + 1,
                        "last_score": signal.score,
                        "coverage_status": status,
                    }
                )
            )
        if not found:
            raise InterviewPlanningError("Previous-answer signal references an unknown competency.")
        return state.model_copy(update={"coverage": coverage_items})

    def _validate_inputs(
        self,
        role: RoleProfile,
        report: SkillMatchReport,
        state: InterviewPlanState,
        previous: PreviousAnswerSignal | None,
    ) -> None:
        role_competencies = {competency.id for competency in role.required_competencies + role.preferred_competencies}
        coverage_ids = {coverage.competency_id for coverage in state.coverage}
        if coverage_ids != role_competencies:
            raise InterviewPlanningError("Coverage competency IDs must match the role profile exactly.")
        history_ids = {question.question_id for question in state.question_history}
        for question in state.question_history:
            if question.competency_id is not None and question.competency_id not in role_competencies:
                raise InterviewPlanningError("Question references a competency absent from the role profile.")
        report_ids = {
            match.competency_id
            for match in report.required_competency_results + report.preferred_competency_results
        }
        if report_ids != role_competencies:
            raise InterviewPlanningError("SkillMatchReport must contain exactly the role competencies.")
        if previous is not None:
            if previous.question_id not in history_ids:
                raise InterviewPlanningError("Previous-answer signal must reference a planned question.")
            if previous.competency_id not in role_competencies:
                raise InterviewPlanningError("Previous-answer signal references an unknown competency.")
            referenced_question = next(
                question for question in state.question_history if question.question_id == previous.question_id
            )
            if referenced_question.competency_id != previous.competency_id:
                raise InterviewPlanningError(
                    "Previous-answer signal competency must match the referenced question."
                )

    @staticmethod
    def _next_question_id(state: InterviewPlanState) -> str:
        return f"q_{state.questions_generated + 1:03d}"

    def _static_plan(
        self,
        state: InterviewPlanState,
        *,
        stage: Literal["introduction", "closing"],
        question_type: Literal["open_ended"],
        difficulty: int,
        objective: str,
        expected_evidence: list[str],
        reason: str,
    ) -> QuestionPlan:
        return QuestionPlan(
            question_id=self._next_question_id(state),
            competency_id=None,
            stage=stage,
            question_type=question_type,
            difficulty=difficulty,
            objective=objective,
            expected_evidence=expected_evidence,
            follow_up_of=None,
            reason=reason,
        )

    @staticmethod
    def _append_plan(state: InterviewPlanState, plan: QuestionPlan) -> InterviewPlanState:
        history = [*state.question_history, plan]
        follow_ups = state.follow_up_count + (1 if plan.question_type == "follow_up" else 0)
        return InterviewPlanState(
            target_question_count=state.target_question_count,
            questions_generated=len(history),
            current_stage=plan.stage,
            coverage=state.coverage,
            follow_up_count=follow_ups,
            max_follow_ups=state.max_follow_ups,
            question_history=history,
        )

    @staticmethod
    def _rank_competencies(
        role: RoleProfile, report: SkillMatchReport, state: InterviewPlanState
    ) -> list[_RankedCompetency]:
        coverage_by_id = {coverage.competency_id: coverage for coverage in state.coverage}
        report_by_id = {
            match.competency_id: match
            for match in report.required_competency_results + report.preferred_competency_results
        }
        ranked: list[_RankedCompetency] = []
        for level, competencies in (
            ("required", role.required_competencies),
            ("preferred", role.preferred_competencies),
        ):
            for competency in competencies:
                ranked.append(
                    _RankedCompetency(
                        competency=competency,
                        match=report_by_id[competency.id],
                        level=level,
                        coverage=coverage_by_id[competency.id],
                    )
                )
        status_rank = {"not_demonstrated": 0, "partial": 1, "strong": 2}
        coverage_rank = {"not_covered": 0, "partial": 1, "covered": 2}
        return sorted(
            ranked,
            key=lambda item: (
                0 if item.level == "required" else 1,
                status_rank[item.match.match_status],
                coverage_rank[item.coverage.coverage_status],
                -item.competency.weight,
                item.competency.id,
            ),
        )

    def _select_target(
        self,
        ranked: list[_RankedCompetency],
        previous: PreviousAnswerSignal | None,
        state: InterviewPlanState,
    ) -> tuple[_RankedCompetency, str | None]:
        if not ranked:
            raise InterviewPlanningError("No competency is available for a non-introduction interview slot.")
        if (
            previous is not None
            and previous.score < 3
            and previous.gaps
            and state.follow_up_count < state.max_follow_ups
        ):
            for item in ranked:
                if item.competency.id == previous.competency_id:
                    return item, previous.question_id
        return ranked[0], None

    @staticmethod
    def _select_format(
        candidate: CandidateProfile,
        selected: _RankedCompetency,
        state: InterviewPlanState,
        is_follow_up: bool,
    ) -> tuple[
        Literal["resume", "technical", "behavioral", "scenario"],
        Literal["technical", "behavioral", "project_deep_dive", "scenario", "follow_up"],
        int,
    ]:
        if is_follow_up:
            return "technical", "follow_up", 3
        non_open_history = [question for question in state.question_history if question.stage not in {"introduction", "closing"}]
        has_behavioral = any(question.stage == "behavioral" for question in non_open_history)
        if not has_behavioral and len(non_open_history) >= 2:
            return "behavioral", "behavioral", 2
        if (candidate.projects or candidate.experiences) and selected.match.match_status in {"strong", "partial"}:
            has_resume = any(question.stage == "resume" for question in non_open_history)
            if not has_resume:
                return "resume", "project_deep_dive", 3
        if len(non_open_history) >= 3:
            return "scenario", "scenario", 4
        difficulty = {"not_demonstrated": 2, "partial": 3, "strong": 4}[selected.match.match_status]
        return "technical", "technical", difficulty

    def _generate_guidance(
        self,
        selected: _RankedCompetency,
        previous: PreviousAnswerSignal | None,
    ) -> QuestionGuidance:
        candidate_evidence = [evidence.model_dump() for evidence in selected.match.candidate_evidence]
        competency_payload = {
            "id": selected.competency.id,
            "name": selected.competency.name,
            "requirements": selected.competency.requirements,
            "job_requirement_evidence": [evidence.model_dump() for evidence in selected.competency.evidence],
        }
        return get_structured_client().generate_structured(
            question_guidance_messages(
                competency_payload,
                candidate_evidence,
                selected.coverage.model_dump(),
                previous.model_dump() if previous is not None else None,
                QuestionGuidance.model_json_schema(),
            ),
            QuestionGuidance,
        )

    @staticmethod
    def _selection_reason(selected: _RankedCompetency, is_follow_up: bool) -> str:
        if is_follow_up:
            return "A low-scoring prior answer exposed a gap and follow-up capacity remains."
        return (
            f"{selected.level.title()} competency ranked by resume-match status "
            f"({selected.match.match_status}), coverage ({selected.coverage.coverage_status}), "
            f"and weight ({selected.competency.weight:g})."
        )
