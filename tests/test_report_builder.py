"""Offline tests for deterministic, evidence-grounded final interview reports."""

from __future__ import annotations

import json
import unittest

from ai.llm_client import configure_structured_client
from ai.report_builder import FinalReportBuilder, ReportBuilderError
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
    InterviewSessionState,
    QuestionPlan,
    RoleProfile,
    SkillMatch,
    SkillMatchReport,
)


class FakeModel:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self._responses = iter(responses)
        self.calls = 0

    def chat(self, messages: list[dict[str, str]]) -> dict[str, object]:
        self.calls += 1
        return {"choices": [{"message": {"content": json.dumps(next(self._responses))}}]}


def competency(competency_id: str, weight: float = 1) -> Competency:
    return Competency(
        id=competency_id,
        name=competency_id.title(),
        requirements=[f"Demonstrate {competency_id}."],
        weight=weight,
        evidence=[Evidence(excerpt=f"Demonstrate {competency_id}.")],
    )


def question(question_id: str, competency_id: str | None = None, follow_up_of: str | None = None) -> InterviewQuestion:
    return InterviewQuestion(
        question_id=question_id,
        question_text=f"Question {question_id}?",
        competency_id=competency_id,
        question_type="follow_up" if follow_up_of else "technical",
        difficulty=2,
        objective="Assess the selected topic.",
        expected_evidence=[],
        follow_up_of=follow_up_of,
    )


def assessment(question_id: str, competency_id: str | None, score: float, excerpt: str | None) -> AnswerAssessment:
    return AnswerAssessment(
        question_id=question_id,
        competency_id=competency_id,
        overall_score=score,
        dimensions=AssessmentDimensions(
            technical_accuracy=score, relevance=score, specificity=score, communication=score
        ),
        evidence=[] if excerpt is None else [{"excerpt": excerpt, "source": "answer"}],
        strengths=["Explained a concrete approach."] if score >= 4 else [],
        gaps=["Needs more detail."] if score < 3 else [],
        recommended_action="advance",
        follow_up_focus=None,
        confidence=1,
    )


def qualitative(excerpt: str, question_id: str = "q_002") -> dict[str, object]:
    finding = {"text": "The answer gave a concrete explanation.", "evidence": [{"question_id": question_id, "excerpt": excerpt, "source": "answer"}]}
    return {
        "strongest_areas": [finding], "weakest_areas": [], "communication_feedback": [],
        "technical_feedback": [], "improvement_recommendations": [], "next_steps": [],
    }


def state(*, required: bool = True, preferred: bool = True, required_score: float = 3, preferred_score: float = 5, status: str = "completed", blank: bool = False) -> InterviewSessionState:
    required_item = competency("python", 2)
    preferred_item = competency("sql", 1)
    role = RoleProfile(
        required_competencies=[required_item] if required else [],
        preferred_competencies=[preferred_item] if preferred else [],
    )
    matches = []
    if required:
        matches.append(SkillMatch(competency_id="python", competency_name="Python", requirement_level="required", competency_weight=2, match_status="partial", candidate_evidence=[Evidence(excerpt="Used Python.")], job_requirement_evidence=required_item.evidence, reasoning="fixture"))
    if preferred:
        matches.append(SkillMatch(competency_id="sql", competency_name="Sql", requirement_level="preferred", competency_weight=1, match_status="partial", candidate_evidence=[Evidence(excerpt="Used SQL.")], job_requirement_evidence=preferred_item.evidence, reasoning="fixture"))
    report = SkillMatchReport(
        required_competency_results=[item for item in matches if item.requirement_level == "required"],
        preferred_competency_results=[item for item in matches if item.requirement_level == "preferred"],
        partial_match_competency_ids=[item.competency_id for item in matches],
    )
    generated = [question("q_001")]
    records = []
    coverage = []
    if required:
        generated.extend([question("q_002", "python"), question("q_003", "python", "q_002")])
        excerpt = None if blank else "I used cross-validation."
        records.extend([
            InterviewAnswerRecord(question_id="q_002", transcript=excerpt or "", assessment=assessment("q_002", "python", required_score, excerpt)),
            InterviewAnswerRecord(question_id="q_003", transcript=excerpt or "", assessment=assessment("q_003", "python", required_score, excerpt)),
        ])
        coverage.append(InterviewCoverage(competency_id="python", questions_asked=2, follow_up_questions=1, last_score=required_score, average_score=required_score, coverage_status="covered" if required_score >= 4 else "partial"))
    if preferred:
        generated.append(question("q_004", "sql"))
        excerpt = None if blank else "I optimized a SQL query."
        records.append(InterviewAnswerRecord(question_id="q_004", transcript=excerpt or "", assessment=assessment("q_004", "sql", preferred_score, excerpt)))
        coverage.append(InterviewCoverage(competency_id="sql", questions_asked=1, last_score=preferred_score, average_score=preferred_score, coverage_status="covered" if preferred_score >= 4 else "partial"))
    plans = [
        QuestionPlan(question_id=item.question_id, competency_id=item.competency_id, stage="technical", question_type=item.question_type, difficulty=item.difficulty, objective=item.objective, expected_evidence=item.expected_evidence, follow_up_of=item.follow_up_of, reason="fixture")
        for item in generated
    ]
    plan_state = InterviewPlanState(target_question_count=len(generated), questions_generated=len(generated), current_stage="closing", coverage=coverage, follow_up_count=sum(item.question_type == "follow_up" for item in generated), question_history=plans)
    return InterviewSessionState(session_id="report-test", status=status, candidate_profile=CandidateProfile(), role_profile=role, skill_match_report=report, plan_state=plan_state, generated_questions=generated, answer_history=records, remaining_questions=0)


class FinalReportBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = FinalReportBuilder()

    def build(self, interview_state: InterviewSessionState, response: dict[str, object] | None = None):
        if response is None:
            first_evidence = next(
                (
                    (record.question_id, item.excerpt)
                    for record in interview_state.answer_history
                    for item in record.assessment.evidence
                ),
                ("q_002", "I used cross-validation."),
            )
            response = qualitative(first_evidence[1], first_evidence[0])
        configure_structured_client(FakeModel([response]))
        return self.builder.build(interview_state)

    def test_multiple_competencies_scores_statistics_and_follow_up_are_deterministic(self) -> None:
        report = self.build(state())
        self.assertEqual([item.interview_score for item in report.competency_results], [60.0, 100.0])
        self.assertEqual(report.job_fit_score, 68.0)
        self.assertEqual(report.overall_score, 73.4)
        self.assertEqual(report.interview_statistics.follow_up_questions, 1)
        self.assertEqual(report.interview_statistics.competencies_assessed, 2)

    def test_required_preferred_weighting_and_required_weakness_cap(self) -> None:
        report = self.build(state(required_score=2, preferred_score=5))
        self.assertEqual(report.job_fit_score, 52.0)
        self.assertEqual(report.recommendation, "weak_match")

    def test_only_required_only_preferred_and_no_competencies(self) -> None:
        self.assertEqual(self.build(state(preferred=False, required_score=4)).job_fit_score, 80.0)
        self.assertEqual(self.build(state(required=False, preferred=True, preferred_score=4)).job_fit_score, 80.0)
        empty = self.build(state(required=False, preferred=False), {"strongest_areas": [], "weakest_areas": [], "communication_feedback": [], "technical_feedback": [], "improvement_recommendations": [], "next_steps": []})
        self.assertEqual(empty.job_fit_score, 0.0)
        self.assertEqual(empty.competency_results, [])

    def test_blank_answers_are_zero_and_do_not_call_llm(self) -> None:
        fake = FakeModel([])
        configure_structured_client(fake)
        report = self.builder.build(state(preferred=False, required_score=0, blank=True))
        self.assertEqual(report.overall_score, 0.0)
        self.assertEqual(report.competency_results[0].interview_score, 0.0)
        self.assertEqual(fake.calls, 0)

    def test_incomplete_state_is_explicitly_marked(self) -> None:
        report = self.build(state(status="active"))
        self.assertEqual(report.status, "incomplete")

    def test_qualitative_content_cannot_change_scores_or_use_fabricated_evidence(self) -> None:
        original = self.build(state())
        response = qualitative("Invented Kubernetes work.")
        with self.assertRaises(ReportBuilderError):
            self.build(state(), response)
        repeated = self.build(state())
        self.assertEqual((original.overall_score, original.job_fit_score), (repeated.overall_score, repeated.job_fit_score))

    def test_resume_context_is_not_interview_evidence_and_transcript_is_verified(self) -> None:
        bad_state = state()
        record = bad_state.answer_history[0].model_copy(update={"transcript": "No Python evidence here."})
        bad_state = bad_state.model_copy(update={"answer_history": [record, *bad_state.answer_history[1:]]})
        with self.assertRaises(ReportBuilderError):
            self.build(bad_state)

    def test_recommendation_thresholds(self) -> None:
        self.assertEqual(self.builder._recommendation(90, 90, 90), "strong_match")
        self.assertEqual(self.builder._recommendation(75, 75, 75), "good_match")
        self.assertEqual(self.builder._recommendation(60, 60, 60), "developing_match")
        self.assertEqual(self.builder._recommendation(59.9, 59.9, None), "weak_match")


if __name__ == "__main__":
    unittest.main()
