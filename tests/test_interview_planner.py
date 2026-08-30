"""Offline tests for deterministic, evidence-aware interview planning."""

from __future__ import annotations

import json
import unittest

from pydantic import ValidationError

from ai.interview_planner import InterviewPlanner, InterviewPlanningError
from ai.llm_client import configure_structured_client
from ai.schemas import (
    CandidateProfile,
    Evidence,
    InterviewCoverage,
    InterviewPlanState,
    PreviousAnswerSignal,
    QuestionPlan,
    RoleProfile,
    Skill,
    SkillMatch,
    SkillMatchReport,
)


class FakeModel:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self._responses = iter(responses)

    def chat(self, messages: list[dict[str, str]]) -> dict[str, object]:
        return {"choices": [{"message": {"content": json.dumps(next(self._responses))}}]}


def guidance(objective: str = "Assess the selected competency.") -> dict[str, object]:
    return {"objective": objective, "expected_evidence": ["A concrete explanation of the approach."]}


def competency(competency_id: str, name: str, weight: float = 1.0) -> dict[str, object]:
    requirement = f"Demonstrate {name}."
    return {
        "id": competency_id,
        "name": name,
        "requirements": [requirement],
        "weight": weight,
        "evidence": [{"excerpt": requirement}],
    }


def match(competency_id: str, name: str, level: str, status: str, weight: float = 1.0) -> SkillMatch:
    requirement = f"Demonstrate {name}."
    evidence = [Evidence(excerpt=f"Used {name} in a project.")] if status != "not_demonstrated" else []
    return SkillMatch(
        competency_id=competency_id,
        competency_name=name,
        requirement_level=level,
        competency_weight=weight,
        match_status=status,
        candidate_evidence=evidence,
        job_requirement_evidence=[Evidence(excerpt=requirement)],
        reasoning="Offline match fixture.",
    )


def report(required: list[SkillMatch], preferred: list[SkillMatch]) -> SkillMatchReport:
    all_matches = required + preferred
    return SkillMatchReport(
        required_competency_results=required,
        preferred_competency_results=preferred,
        matched_competency_ids=[item.competency_id for item in all_matches if item.match_status == "strong"],
        partial_match_competency_ids=[item.competency_id for item in all_matches if item.match_status == "partial"],
        not_demonstrated_competency_ids=[
            item.competency_id for item in all_matches if item.match_status == "not_demonstrated"
        ],
    )


class InterviewPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = InterviewPlanner()
        self.candidate = CandidateProfile(
            skills=[Skill(name="Python", evidence=[Evidence(excerpt="Used Python in a project.")])]
        )

    def _start(self, role: RoleProfile, matches: SkillMatchReport, target: int = 5, max_follow_ups: int = 2):
        state = self.planner.create_initial_state(role, target, max_follow_ups)
        introduction, state = self.planner.plan_next(self.candidate, role, matches, state)
        return introduction, state

    def test_introduction_is_selected_at_the_beginning(self) -> None:
        role = RoleProfile(required_competencies=[competency("python", "Python")])
        initial = self.planner.create_initial_state(role, 3)

        question, state = self.planner.plan_next(self.candidate, role, report([match("python", "Python", "required", "strong")], []), initial)

        self.assertEqual(question.stage, "introduction")
        self.assertIsNone(question.competency_id)
        self.assertEqual(state.questions_generated, 1)

    def test_required_not_demonstrated_competency_is_prioritized(self) -> None:
        role = RoleProfile(
            required_competencies=[competency("docker", "Docker", 2.0)],
            preferred_competencies=[competency("python", "Python", 5.0)],
        )
        matches = report(
            [match("docker", "Docker", "required", "not_demonstrated", 2.0)],
            [match("python", "Python", "preferred", "strong", 5.0)],
        )
        _, state = self._start(role, matches)
        configure_structured_client(FakeModel([guidance(), guidance()]))

        question, _ = self.planner.plan_next(self.candidate, role, matches, state)

        self.assertEqual(question.competency_id, "docker")

    def test_required_partial_is_prioritized_over_preferred_competency(self) -> None:
        role = RoleProfile(
            required_competencies=[competency("warehouse", "Data warehouse")],
            preferred_competencies=[competency("python", "Python")],
        )
        matches = report(
            [match("warehouse", "Data warehouse", "required", "partial")],
            [match("python", "Python", "preferred", "not_demonstrated")],
        )
        _, state = self._start(role, matches)
        configure_structured_client(FakeModel([guidance()]))

        question, _ = self.planner.plan_next(self.candidate, role, matches, state)

        self.assertEqual(question.competency_id, "warehouse")

    def test_strong_competencies_are_still_covered(self) -> None:
        role = RoleProfile(required_competencies=[competency("python", "Python")])
        matches = report([match("python", "Python", "required", "strong")], [])
        _, state = self._start(role, matches)
        configure_structured_client(FakeModel([guidance()]))

        question, _ = self.planner.plan_next(self.candidate, role, matches, state)

        self.assertEqual(question.competency_id, "python")

    def test_already_covered_competencies_receive_lower_priority(self) -> None:
        role = RoleProfile(required_competencies=[competency("docker", "Docker"), competency("python", "Python")])
        matches = report(
            [match("docker", "Docker", "required", "not_demonstrated"), match("python", "Python", "required", "not_demonstrated")],
            [],
        )
        _, state = self._start(role, matches)
        state = InterviewPlanState(
            target_question_count=state.target_question_count,
            questions_generated=state.questions_generated,
            current_stage=state.current_stage,
            coverage=[
                InterviewCoverage(competency_id="docker", questions_asked=1, last_score=5, coverage_status="covered"),
                InterviewCoverage(competency_id="python"),
            ],
            follow_up_count=state.follow_up_count,
            max_follow_ups=state.max_follow_ups,
            question_history=state.question_history,
        )
        configure_structured_client(FakeModel([guidance()]))

        question, _ = self.planner.plan_next(self.candidate, role, matches, state)

        self.assertEqual(question.competency_id, "python")

    def test_low_scoring_previous_answer_triggers_follow_up(self) -> None:
        role = RoleProfile(required_competencies=[competency("python", "Python")])
        matches = report([match("python", "Python", "required", "partial")], [])
        _, state = self._start(role, matches)
        configure_structured_client(FakeModel([guidance("Assess Python approach."), guidance("Assess validation strategy.")]))
        first_competency_question, state = self.planner.plan_next(self.candidate, role, matches, state)
        signal = PreviousAnswerSignal(
            question_id=first_competency_question.question_id,
            competency_id="python",
            score=2,
            gaps=["The candidate did not explain model validation."],
        )

        question, _ = self.planner.plan_next(self.candidate, role, matches, state, signal)

        self.assertEqual(question.question_type, "follow_up")
        self.assertEqual(question.follow_up_of, first_competency_question.question_id)

    def test_follow_up_count_cannot_exceed_maximum(self) -> None:
        role = RoleProfile(required_competencies=[competency("python", "Python")])
        matches = report([match("python", "Python", "required", "partial")], [])
        _, state = self._start(role, matches, max_follow_ups=0)
        configure_structured_client(FakeModel([guidance(), guidance()]))
        first_question, state = self.planner.plan_next(self.candidate, role, matches, state)
        signal = PreviousAnswerSignal(question_id=first_question.question_id, competency_id="python", score=1, gaps=["Missing validation."])

        question, _ = self.planner.plan_next(self.candidate, role, matches, state, signal)

        self.assertNotEqual(question.question_type, "follow_up")

    def test_target_count_is_respected_and_closing_uses_final_slot(self) -> None:
        role = RoleProfile(required_competencies=[competency("python", "Python")])
        matches = report([match("python", "Python", "required", "strong")], [])
        _, state = self._start(role, matches, target=2)

        closing, state = self.planner.plan_next(self.candidate, role, matches, state)
        after_completion, final_state = self.planner.plan_next(self.candidate, role, matches, state)

        self.assertEqual(closing.stage, "closing")
        self.assertEqual(state.questions_generated, 2)
        self.assertIsNone(after_completion)
        self.assertEqual(final_state.current_stage, "closing")

    def test_invalid_competency_references_are_rejected(self) -> None:
        role = RoleProfile(required_competencies=[competency("python", "Python")])
        matches = report([match("python", "Python", "required", "strong")], [])
        invalid_state = InterviewPlanState(target_question_count=3, coverage=[])

        with self.assertRaises(InterviewPlanningError):
            self.planner.plan_next(self.candidate, role, matches, invalid_state)

    def test_difficulty_is_always_within_schema_bounds(self) -> None:
        role = RoleProfile(required_competencies=[competency("python", "Python")])
        matches = report([match("python", "Python", "required", "strong")], [])
        _, state = self._start(role, matches)
        configure_structured_client(FakeModel([guidance()]))

        question, _ = self.planner.plan_next(self.candidate, role, matches, state)

        self.assertGreaterEqual(question.difficulty, 1)
        self.assertLessEqual(question.difficulty, 5)

    def test_empty_competency_lists_do_not_crash(self) -> None:
        role = RoleProfile()
        matches = report([], [])
        _, state = self._start(role, matches, target=3)

        question, state = self.planner.plan_next(self.candidate, role, matches, state)
        closing, _ = self.planner.plan_next(self.candidate, role, matches, state)

        self.assertEqual(question.stage, "behavioral")
        self.assertIsNone(question.competency_id)
        self.assertEqual(closing.stage, "closing")

    def test_resume_match_and_interview_coverage_remain_separate(self) -> None:
        role = RoleProfile(required_competencies=[competency("python", "Python")])
        matches = report([match("python", "Python", "required", "strong")], [])
        _, state = self._start(role, matches)
        configure_structured_client(FakeModel([guidance()]))

        question, updated_state = self.planner.plan_next(self.candidate, role, matches, state)

        self.assertEqual(question.competency_id, "python")
        self.assertEqual(updated_state.coverage[0].coverage_status, "not_covered")

    def test_deterministic_ranking_is_stable(self) -> None:
        role = RoleProfile(required_competencies=[competency("zeta", "Zeta"), competency("alpha", "Alpha")])
        matches = report(
            [match("zeta", "Zeta", "required", "partial"), match("alpha", "Alpha", "required", "partial")],
            [],
        )
        _, first_state = self._start(role, matches)
        _, second_state = self._start(role, matches)
        configure_structured_client(FakeModel([guidance(), guidance()]))

        first_question, _ = self.planner.plan_next(self.candidate, role, matches, first_state)
        second_question, _ = self.planner.plan_next(self.candidate, role, matches, second_state)

        self.assertEqual(first_question.competency_id, "alpha")
        self.assertEqual(first_question.competency_id, second_question.competency_id)

    def test_state_rejects_invalid_follow_up_references(self) -> None:
        with self.assertRaises(ValidationError):
            InterviewPlanState(
                target_question_count=2,
                questions_generated=1,
                follow_up_count=1,
                question_history=[
                    QuestionPlan(
                        question_id="q_001",
                        competency_id="python",
                        stage="technical",
                        question_type="follow_up",
                        difficulty=3,
                        objective="Clarify validation.",
                        expected_evidence=[],
                        follow_up_of="q_missing",
                        reason="A gap was found.",
                    )
                ],
            )


if __name__ == "__main__":
    unittest.main()
