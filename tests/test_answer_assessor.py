"""Offline tests for evidence-grounded assessment of one interview answer."""

from __future__ import annotations

import json
import unittest

from pydantic import ValidationError

from ai.answer_assessor import AnswerAssessmentError, AnswerAssessor
from ai.llm_client import configure_structured_client
from ai.schemas import (
    AnswerAssessmentContent,
    Competency,
    Evidence,
    InterviewQuestion,
    QuestionPlan,
)


class FakeModel:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self._responses = iter(responses)
        self.calls = 0

    def chat(self, messages: list[dict[str, str]]) -> dict[str, object]:
        self.calls += 1
        return {"choices": [{"message": {"content": json.dumps(next(self._responses))}}]}


def question() -> InterviewQuestion:
    return InterviewQuestion(
        question_id="q_002",
        question_text="How did you validate your machine learning model?",
        competency_id="ml_validation",
        question_type="technical",
        difficulty=3,
        objective="Assess model validation strategy.",
        expected_evidence=["Metric choice and validation approach."],
        follow_up_of=None,
    )


def competency() -> Competency:
    requirement = "Validate machine learning models using appropriate metrics."
    return Competency(
        id="ml_validation",
        name="Model validation",
        requirements=[requirement],
        weight=2.0,
        evidence=[Evidence(excerpt=requirement)],
    )


def assessment(
    *,
    score: float = 4,
    evidence_excerpt: str | None = None,
    action: str = "advance",
    focus: str | None = None,
    strengths: list[str] | None = None,
    gaps: list[str] | None = None,
    confidence: float = 0.8,
) -> dict[str, object]:
    return {
        "overall_score": score,
        "technical_accuracy": score,
        "relevance": score,
        "specificity": score,
        "communication": score,
        "evidence": ([{"excerpt": evidence_excerpt, "source": "answer"}] if evidence_excerpt else []),
        "strengths": strengths or ["Explains a concrete validation approach."],
        "gaps": gaps or [],
        "recommended_action": action,
        "follow_up_focus": focus,
        "confidence": confidence,
    }


class AnswerAssessorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assessor = AnswerAssessor()
        self.question = question()
        self.competency = competency()
        self.job_evidence = self.competency.evidence
        self.candidate_evidence = [Evidence(excerpt="Built ML pipelines using Python.")]

    def _assess(self, transcript: str, response: dict[str, object]):
        configure_structured_client(FakeModel([response]))
        return self.assessor.assess(
            self.question,
            transcript,
            self.competency,
            self.job_evidence,
            self.candidate_evidence,
        )

    def test_excellent_technical_answer(self) -> None:
        transcript = "I used cross-validation and compared precision-recall with ROC-AUC before deployment."
        result = self._assess(transcript, assessment(score=5, evidence_excerpt="I used cross-validation"))

        self.assertEqual(result.overall_score, 5)
        self.assertEqual(result.evidence[0].source, "answer")

    def test_weak_technical_answer(self) -> None:
        transcript = "I used accuracy."
        result = self._assess(
            transcript,
            assessment(
                score=1,
                evidence_excerpt="I used accuracy.",
                action="follow_up",
                focus="Ask why accuracy was selected and whether other metrics were considered.",
                gaps=["The answer does not justify the metric choice."],
            ),
        )

        self.assertEqual(result.recommended_action, "follow_up")
        self.assertIsNotNone(result.follow_up_focus)

    def test_relevant_but_insufficient_answer(self) -> None:
        transcript = "I checked the model before deployment."
        result = self._assess(
            transcript,
            assessment(
                score=2,
                evidence_excerpt="I checked the model before deployment.",
                action="follow_up",
                focus="Ask which validation method and metrics were used.",
            ),
        )

        self.assertEqual(result.dimensions.specificity, 2)

    def test_strong_behavioral_answer_is_supported(self) -> None:
        behavioral_question = self.question.model_copy(
            update={"question_type": "behavioral", "question_text": "Tell me about a validation trade-off you made."}
        )
        transcript = "I explained the precision-recall trade-off to stakeholders and documented the decision."
        configure_structured_client(FakeModel([assessment(score=4, evidence_excerpt="I explained the precision-recall trade-off")]))

        result = self.assessor.assess(behavioral_question, transcript, self.competency, self.job_evidence)

        self.assertEqual(result.overall_score, 4)

    def test_project_answer_evidence_is_grounded_in_transcript(self) -> None:
        transcript = "For the forecasting project, I used time-series cross-validation to prevent leakage."
        result = self._assess(
            transcript,
            assessment(score=4, evidence_excerpt="I used time-series cross-validation to prevent leakage."),
        )

        self.assertIn(result.evidence[0].excerpt, transcript)

    def test_empty_answer_is_deterministic_and_skips_llm(self) -> None:
        fake = FakeModel([])
        configure_structured_client(fake)

        result = self.assessor.assess(self.question, "   \n\t", self.competency, self.job_evidence)

        self.assertEqual(result.overall_score, 0)
        self.assertEqual(result.evidence, [])
        self.assertEqual(result.gaps, ["No substantive answer was provided."])
        self.assertEqual(result.confidence, 1.0)
        self.assertEqual(fake.calls, 0)

    def test_fabricated_evidence_is_rejected(self) -> None:
        transcript = "I used accuracy."
        configure_structured_client(FakeModel([assessment(evidence_excerpt="I used cross-validation.")]))

        with self.assertRaises(AnswerAssessmentError):
            self.assessor.assess(self.question, transcript, self.competency, self.job_evidence)

    def test_resume_only_evidence_cannot_be_used_as_answer_evidence(self) -> None:
        transcript = "I worked mainly with SQL."
        configure_structured_client(FakeModel([assessment(evidence_excerpt="Built ML pipelines using Python.")]))

        with self.assertRaises(AnswerAssessmentError):
            self.assessor.assess(
                self.question,
                transcript,
                self.competency,
                self.job_evidence,
                self.candidate_evidence,
            )

    def test_follow_up_focus_schema_rules(self) -> None:
        with self.assertRaises(ValidationError):
            AnswerAssessmentContent.model_validate(assessment(action="follow_up", focus=None))
        with self.assertRaises(ValidationError):
            AnswerAssessmentContent.model_validate(assessment(action="advance", focus="Ask more."))

    def test_question_identity_and_competency_are_copied_deterministically(self) -> None:
        result = self._assess("I used cross-validation.", assessment(evidence_excerpt="I used cross-validation."))

        self.assertEqual(result.question_id, self.question.question_id)
        self.assertEqual(result.competency_id, self.question.competency_id)

    def test_score_and_confidence_bounds_are_enforced(self) -> None:
        with self.assertRaises(ValidationError):
            AnswerAssessmentContent.model_validate(assessment(score=6))
        with self.assertRaises(ValidationError):
            AnswerAssessmentContent.model_validate(assessment(confidence=1.1))

    def test_repeated_model_output_is_deterministic(self) -> None:
        transcript = "I used cross-validation."
        response = assessment(evidence_excerpt="I used cross-validation.")
        configure_structured_client(FakeModel([response, response]))

        first = self.assessor.assess(self.question, transcript, self.competency, self.job_evidence)
        second = self.assessor.assess(self.question, transcript, self.competency, self.job_evidence)

        self.assertEqual(first, second)

    def test_internal_instruction_text_is_rejected(self) -> None:
        transcript = "I used cross-validation."
        configure_structured_client(
            FakeModel([assessment(evidence_excerpt="I used cross-validation.", strengths=["The planner selected this answer."])])
        )

        with self.assertRaises(AnswerAssessmentError):
            self.assessor.assess(self.question, transcript, self.competency, self.job_evidence)

    def test_question_plan_must_match_question_metadata(self) -> None:
        mismatched_plan = QuestionPlan(
            question_id="q_other",
            competency_id="ml_validation",
            stage="technical",
            question_type="technical",
            difficulty=3,
            objective="Assess model validation strategy.",
            expected_evidence=["Metric choice and validation approach."],
            follow_up_of=None,
            reason="Test fixture.",
        )

        with self.assertRaises(AnswerAssessmentError):
            self.assessor.assess(
                self.question,
                "I used cross-validation.",
                self.competency,
                self.job_evidence,
                question_plan=mismatched_plan,
            )


if __name__ == "__main__":
    unittest.main()
