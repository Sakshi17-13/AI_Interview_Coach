"""Offline integration tests for the session-local adaptive interview orchestrator."""

from __future__ import annotations

import unittest

from ai.interview_engine import InterviewEngine, InterviewEngineError
from ai.schemas import (
    AnswerAssessment,
    AssessmentDimensions,
    CandidateProfile,
    Competency,
    Evidence,
    InterviewAnswerRecord,
    InterviewCoverage,
    InterviewPlanState,
    InterviewQuestion,
    PreviousAnswerSignal,
    QuestionPlan,
    RoleProfile,
    SkillMatch,
    SkillMatchReport,
)


def competency(competency_id: str, weight: float = 1.0) -> Competency:
    return Competency(
        id=competency_id,
        name=competency_id.title(),
        requirements=[f"Demonstrate {competency_id}."],
        weight=weight,
        evidence=[Evidence(excerpt=f"Demonstrate {competency_id}.")],
    )


def match(item: Competency, level: str = "required") -> SkillMatch:
    return SkillMatch(
        competency_id=item.id,
        competency_name=item.name,
        requirement_level=level,
        competency_weight=item.weight,
        match_status="partial",
        candidate_evidence=[Evidence(excerpt=f"Candidate used {item.id}.")],
        job_requirement_evidence=item.evidence,
        reasoning="Fixture.",
    )


def profiles() -> tuple[CandidateProfile, RoleProfile, SkillMatchReport]:
    required = competency("required_skill", 2.0)
    preferred = competency("preferred_skill", 1.0)
    role = RoleProfile(required_competencies=[required], preferred_competencies=[preferred])
    required_match, preferred_match = match(required), match(preferred, "preferred")
    report = SkillMatchReport(
        required_competency_results=[required_match],
        preferred_competency_results=[preferred_match],
        partial_match_competency_ids=[required.id, preferred.id],
    )
    return CandidateProfile(), role, report


def make_assessment(question: InterviewQuestion, score: float = 4, action: str = "advance") -> AnswerAssessment:
    return AnswerAssessment(
        question_id=question.question_id,
        competency_id=question.competency_id,
        overall_score=score,
        dimensions=AssessmentDimensions(
            technical_accuracy=score, relevance=score, specificity=score, communication=score
        ),
        evidence=[],
        strengths=[],
        gaps=["Explain the validation choice."] if action == "follow_up" else [],
        recommended_action=action,
        follow_up_focus="Explain the validation choice." if action == "follow_up" else None,
        confidence=1.0,
    )


class FakePlanner:
    def __init__(self, fail_on_plan_call: int | None = None) -> None:
        self.plan_calls = 0
        self.fail_on_plan_call = fail_on_plan_call

    def create_initial_state(self, role: RoleProfile, target: int, max_follow_ups: int) -> InterviewPlanState:
        return InterviewPlanState(
            target_question_count=target,
            max_follow_ups=max_follow_ups,
            coverage=[InterviewCoverage(competency_id=item.id) for item in role.required_competencies + role.preferred_competencies],
        )

    def plan_next(self, candidate, role, report, state, previous: PreviousAnswerSignal | None):
        self.plan_calls += 1
        if self.fail_on_plan_call == self.plan_calls:
            raise RuntimeError("planner failed")
        if state.questions_generated >= state.target_question_count:
            return None, state.model_copy(update={"current_stage": "closing"})
        number = state.questions_generated + 1
        if number == 1:
            plan = QuestionPlan(
                question_id="q_001", competency_id=None, stage="introduction", question_type="open_ended",
                difficulty=1, objective="Introduce the candidate.", expected_evidence=[], follow_up_of=None, reason="intro",
            )
        elif number == state.target_question_count:
            plan = QuestionPlan(
                question_id=f"q_{number:03d}", competency_id=None, stage="closing", question_type="open_ended",
                difficulty=1, objective="Close the interview.", expected_evidence=[], follow_up_of=None, reason="closing",
            )
        else:
            all_competencies = role.required_competencies + role.preferred_competencies
            coverage = {item.competency_id: item for item in state.coverage}
            chosen = next((item for item in role.required_competencies if coverage[item.id].coverage_status != "covered"), None)
            chosen = chosen or next((item for item in all_competencies if coverage[item.id].coverage_status != "covered"), all_competencies[0])
            follow = previous is not None and previous.score < 3 and previous.gaps and state.follow_up_count < state.max_follow_ups
            plan = QuestionPlan(
                question_id=f"q_{number:03d}", competency_id=chosen.id, stage="technical",
                question_type="follow_up" if follow else "technical", difficulty=3,
                objective=f"Assess {chosen.name}.", expected_evidence=["Concrete explanation."],
                follow_up_of=previous.question_id if follow else None, reason="fake deterministic planner",
            )
        history = [*state.question_history, plan]
        updated = InterviewPlanState(
            target_question_count=state.target_question_count,
            questions_generated=len(history), current_stage=plan.stage, coverage=state.coverage,
            follow_up_count=state.follow_up_count + int(plan.question_type == "follow_up"),
            max_follow_ups=state.max_follow_ups, question_history=history,
        )
        return plan, updated

    def record_coverage(self, state: InterviewPlanState, signal: PreviousAnswerSignal, was_follow_up: bool = False):
        coverage = []
        for item in state.coverage:
            if item.competency_id == signal.competency_id:
                count = item.questions_asked + 1
                coverage.append(item.model_copy(update={
                    "questions_asked": count,
                    "follow_up_questions": item.follow_up_questions + int(was_follow_up),
                    "last_score": signal.score,
                    "average_score": round(((item.average_score or 0) * item.questions_asked + signal.score) / count, 2),
                    "coverage_status": "covered" if signal.score >= 4 else "partial",
                }))
            else:
                coverage.append(item)
        return state.model_copy(update={"coverage": coverage})


class FakeQuestionGenerator:
    def __init__(self, fail_on_call: int | None = None) -> None:
        self.calls = 0
        self.fail_on_call = fail_on_call

    def generate(self, plan, competency=None, job_evidence=None, candidate_evidence=None, previous_question=None, previous_answer_signal=None):
        self.calls += 1
        if self.fail_on_call == self.calls:
            raise RuntimeError("question generation failed")
        return InterviewQuestion(
            question_id=plan.question_id, question_text=f"Question {plan.question_id}?",
            competency_id=plan.competency_id, question_type=plan.question_type, difficulty=plan.difficulty,
            objective=plan.objective, expected_evidence=plan.expected_evidence, follow_up_of=plan.follow_up_of,
        )


class FakeAnswerAssessor:
    def __init__(self, outcomes: list[tuple[float, str]]) -> None:
        self.outcomes = iter(outcomes)
        self.transcripts: list[str] = []

    def assess(self, question, transcript, competency=None, job_evidence=None, candidate_evidence=None, plan=None):
        self.transcripts.append(transcript)
        score, action = next(self.outcomes)
        return make_assessment(question, score, action)


class InterviewEngineTests(unittest.TestCase):
    def make_engine(self, outcomes=[(4, "advance"), (4, "advance")], **kwargs):
        planner = kwargs.pop("planner", FakePlanner())
        generator = kwargs.pop("generator", FakeQuestionGenerator())
        assessor = kwargs.pop("assessor", FakeAnswerAssessor(outcomes))
        return InterviewEngine(planner, generator, assessor), planner, generator, assessor

    def test_initialization_generates_first_question_and_isolated_session(self) -> None:
        candidate, role, report = profiles()
        engine, _, _, _ = self.make_engine()

        first = engine.initialize_interview(candidate, role, report, 3, session_id="session-a")

        self.assertEqual(first.question_id, "q_001")
        self.assertEqual(engine.get_state().session_id, "session-a")
        self.assertEqual(engine.get_state().current_question, first)

    def test_submission_stores_assessment_updates_coverage_and_preserves_transcript(self) -> None:
        candidate, role, report = profiles()
        engine, _, _, assessor = self.make_engine([(4, "advance"), (4, "advance")])
        engine.initialize_interview(candidate, role, report, 3)
        result = engine.submit_answer("  exact transcript  ")
        result = engine.submit_answer("second answer")

        state = engine.get_state()
        self.assertEqual(assessor.transcripts[0], "  exact transcript  ")
        self.assertEqual(state.answer_history[0].transcript, "  exact transcript  ")
        self.assertEqual(len(engine.get_assessments()), 2)
        self.assertEqual(state.plan_state.coverage[0].questions_asked, 1)
        self.assertIsNotNone(result.next_question)

    def test_weak_answer_causes_follow_up_and_limit_is_enforced(self) -> None:
        candidate, role, report = profiles()
        engine, _, _, _ = self.make_engine([(4, "advance"), (1, "follow_up"), (1, "follow_up")])
        engine.initialize_interview(candidate, role, report, 5, max_follow_ups=1)
        engine.submit_answer("intro")
        follow_up_result = engine.submit_answer("weak")
        self.assertEqual(follow_up_result.next_question.question_type, "follow_up")
        final_result = engine.submit_answer("still weak")
        self.assertNotEqual(final_result.next_question.question_type, "follow_up")
        self.assertEqual(engine.get_state().plan_state.follow_up_count, 1)

    def test_strong_answer_moves_to_other_competency_and_covered_is_not_repeated(self) -> None:
        candidate, role, report = profiles()
        engine, _, _, _ = self.make_engine([(4, "advance"), (5, "advance")])
        engine.initialize_interview(candidate, role, report, 4)
        engine.submit_answer("intro")
        first_competency = engine.get_current_question()
        engine.submit_answer("strong")
        second_competency = engine.get_current_question()

        self.assertEqual(first_competency.competency_id, "required_skill")
        self.assertEqual(second_competency.competency_id, "preferred_skill")

    def test_target_count_completion_rejects_more_answers(self) -> None:
        candidate, role, report = profiles()
        engine, _, _, _ = self.make_engine([(4, "advance")])
        engine.initialize_interview(candidate, role, report, 1)
        result = engine.submit_answer("intro")

        self.assertIsNone(result.next_question)
        self.assertTrue(engine.is_complete())
        self.assertIsNone(engine.get_current_question())
        with self.assertRaises(InterviewEngineError):
            engine.submit_answer("too late")

    def test_empty_transcript_is_forwarded_exactly_to_assessor(self) -> None:
        candidate, role, report = profiles()
        engine, _, _, assessor = self.make_engine([(0, "change_topic"), (4, "advance")])
        engine.initialize_interview(candidate, role, report, 3)
        engine.submit_answer("")

        self.assertEqual(assessor.transcripts, [""])

    def test_missing_current_question_and_uninitialized_engine_are_rejected(self) -> None:
        engine, _, _, _ = self.make_engine()
        with self.assertRaises(InterviewEngineError):
            engine.get_current_question()
        candidate, role, report = profiles()
        engine.initialize_interview(candidate, role, report, 3)
        engine._state = engine.get_state().model_copy(update={"current_question": None})
        with self.assertRaises(InterviewEngineError):
            engine.submit_answer("answer")

    def test_planner_failure_preserves_completed_assessment(self) -> None:
        candidate, role, report = profiles()
        planner = FakePlanner(fail_on_plan_call=2)
        engine, _, _, _ = self.make_engine([(4, "advance")], planner=planner)
        engine.initialize_interview(candidate, role, report, 3)
        with self.assertRaises(InterviewEngineError):
            engine.submit_answer("intro")

        self.assertEqual(len(engine.get_assessments()), 1)
        self.assertIsNone(engine.get_current_question())
        self.assertFalse(engine.is_complete())

    def test_question_generation_failure_preserves_completed_assessment(self) -> None:
        candidate, role, report = profiles()
        generator = FakeQuestionGenerator(fail_on_call=2)
        engine, _, _, _ = self.make_engine([(4, "advance")], generator=generator)
        engine.initialize_interview(candidate, role, report, 3)
        with self.assertRaises(InterviewEngineError):
            engine.submit_answer("intro")

        self.assertEqual(len(engine.get_assessments()), 1)
        self.assertIsNone(engine.get_current_question())
        self.assertFalse(engine.is_complete())

    def test_two_engines_are_isolated_and_metadata_is_planner_owned(self) -> None:
        candidate, role, report = profiles()
        first, _, _, _ = self.make_engine()
        second, _, _, _ = self.make_engine()
        first_question = first.initialize_interview(candidate, role, report, 2, session_id="one")
        second_question = second.initialize_interview(candidate, role, report, 2, session_id="two")
        first.submit_answer("one answer")

        self.assertEqual(first_question.question_id, second_question.question_id)
        self.assertEqual(first.get_state().session_id, "one")
        self.assertEqual(second.get_state().session_id, "two")
        self.assertEqual(len(second.get_assessments()), 0)
        self.assertEqual(first.get_current_question().question_id, "q_002")

    def test_required_priority_and_determinism(self) -> None:
        candidate, role, report = profiles()
        first, _, _, _ = self.make_engine()
        second, _, _, _ = self.make_engine()
        first.initialize_interview(candidate, role, report, 3)
        second.initialize_interview(candidate, role, report, 3)
        a = first.submit_answer("intro").next_question
        b = second.submit_answer("intro").next_question

        self.assertEqual(a.competency_id, "required_skill")
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
