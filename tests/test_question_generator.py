"""Offline tests for planner-controlled interview question generation."""

from __future__ import annotations

import json
import unittest

from ai.llm_client import configure_structured_client
from ai.question_generator import QuestionGenerationError, QuestionGenerator
from ai.schemas import Competency, Evidence, PreviousAnswerSignal, QuestionPlan


class FakeModel:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self._responses = iter(responses)

    def chat(self, messages: list[dict[str, str]]) -> dict[str, object]:
        return {"choices": [{"message": {"content": json.dumps(next(self._responses))}}]}


def plan(
    *,
    question_type: str = "technical",
    competency_id: str | None = "python",
    difficulty: int = 3,
    follow_up_of: str | None = None,
) -> QuestionPlan:
    return QuestionPlan(
        question_id="q_002",
        competency_id=competency_id,
        stage="technical" if question_type != "behavioral" else "behavioral",
        question_type=question_type,
        difficulty=difficulty,
        objective="Assess Python reasoning and validation trade-offs.",
        expected_evidence=["A concrete validation approach."],
        follow_up_of=follow_up_of,
        reason="Planner-owned selection reason.",
    )


def python_competency() -> Competency:
    return Competency(
        id="python",
        name="Python programming",
        requirements=["Develop and validate Python data pipelines."],
        weight=2.0,
        evidence=[Evidence(excerpt="Develop and validate Python data pipelines.")],
    )


class QuestionGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = QuestionGenerator()
        self.competency = python_competency()
        self.job_evidence = self.competency.evidence
        self.candidate_evidence = [Evidence(excerpt="Developed machine learning pipelines using Python.")]

    def _generate(self, question_plan: QuestionPlan, response: dict[str, object], **kwargs):
        configure_structured_client(FakeModel([response]))
        return self.generator.generate(
            question_plan,
            selected_competency=self.competency if question_plan.competency_id else None,
            job_requirement_evidence=self.job_evidence if question_plan.competency_id else [],
            candidate_evidence=self.candidate_evidence if question_plan.competency_id else [],
            **kwargs,
        )

    def test_normal_technical_question_generation(self) -> None:
        question = self._generate(
            plan(),
            {"question_text": "How would you validate a Python data pipeline before deployment?"},
        )

        self.assertEqual(question.question_type, "technical")
        self.assertEqual(question.question_text, "How would you validate a Python data pipeline before deployment?")

    def test_behavioral_question_generation(self) -> None:
        behavioral_plan = plan(question_type="behavioral", difficulty=2)
        question = self._generate(
            behavioral_plan,
            {"question_text": "Tell me about a time you balanced delivery speed with validation quality?"},
        )

        self.assertEqual(question.question_type, "behavioral")

    def test_project_deep_dive_uses_supplied_candidate_evidence(self) -> None:
        project_evidence = [Evidence(excerpt="Built the Customer Segmentation Model using Python.")]
        project_plan = plan(question_type="project_deep_dive")
        configure_structured_client(
            FakeModel([{"question_text": "How did you validate the Customer Segmentation Model?"}])
        )

        question = self.generator.generate(
            project_plan,
            self.competency,
            self.job_evidence,
            project_evidence,
        )

        self.assertIn("Customer Segmentation Model", question.question_text)

    def test_scenario_question_generation(self) -> None:
        scenario_plan = plan(question_type="scenario", difficulty=4)
        configure_structured_client(
            FakeModel([{"question_text": "How would you design validation for a Python data pipeline with unreliable source data?"}])
        )

        question = self.generator.generate(scenario_plan, self.competency, self.job_evidence, [])

        self.assertEqual(question.question_type, "scenario")

    def test_follow_up_uses_previous_answer_gap(self) -> None:
        follow_up_plan = plan(question_type="follow_up", follow_up_of="q_001")
        signal = PreviousAnswerSignal(
            question_id="q_001", competency_id="python", score=2, gaps=["Model validation was not explained."]
        )
        question = self._generate(
            follow_up_plan,
            {"question_text": "What validation strategy would you use to check the pipeline before deployment?"},
            previous_question="How did you build the pipeline?",
            previous_answer_signal=signal,
        )

        self.assertEqual(question.follow_up_of, "q_001")

    def test_metadata_remains_planner_controlled_when_model_returns_extra_fields(self) -> None:
        valid_text = "How would you validate a Python data pipeline before deployment?"
        configure_structured_client(
            FakeModel([
                {"question_text": valid_text, "competency_id": "docker", "difficulty": 5},
                {"question_text": valid_text},
            ])
        )

        question = self.generator.generate(plan(difficulty=3), self.competency, self.job_evidence, self.candidate_evidence)

        self.assertEqual(question.competency_id, "python")
        self.assertEqual(question.difficulty, 3)

    def test_fabricated_candidate_details_are_rejected(self) -> None:
        configure_structured_client(
            FakeModel([{"question_text": "Could you describe your Kubernetes leadership at Example Corp?"}])
        )

        with self.assertRaises(QuestionGenerationError):
            self.generator.generate(plan(), self.competency, self.job_evidence, self.candidate_evidence)

    def test_empty_question_text_is_rejected(self) -> None:
        configure_structured_client(FakeModel([{"question_text": " "}, {"question_text": " "}]))

        with self.assertRaises(QuestionGenerationError):
            self.generator.generate(plan(), self.competency, self.job_evidence, self.candidate_evidence)

    def test_plan_metadata_is_copied_exactly(self) -> None:
        question_plan = plan(question_type="follow_up", follow_up_of="q_001", difficulty=3)
        signal = PreviousAnswerSignal(question_id="q_001", competency_id="python", score=2, gaps=["Validation gap."])
        question = self._generate(
            question_plan,
            {"question_text": "How would you validate the pipeline before deployment?"},
            previous_question="How did you build the pipeline?",
            previous_answer_signal=signal,
        )

        self.assertEqual(question.question_id, question_plan.question_id)
        self.assertEqual(question.expected_evidence, question_plan.expected_evidence)
        self.assertEqual(question.follow_up_of, question_plan.follow_up_of)
        self.assertEqual(question.objective, question_plan.objective)

    def test_question_cannot_expose_internal_metadata_or_multiple_questions(self) -> None:
        configure_structured_client(
            FakeModel([{"question_text": "Because your resume says Python, what is your competency ID?"}])
        )

        with self.assertRaises(QuestionGenerationError):
            self.generator.generate(plan(), self.competency, self.job_evidence, self.candidate_evidence)

        configure_structured_client(
            FakeModel([{"question_text": "How would you validate the pipeline? What trade-off matters most?"}])
        )
        with self.assertRaises(QuestionGenerationError):
            self.generator.generate(plan(), self.competency, self.job_evidence, self.candidate_evidence)

    def test_generation_is_deterministic_for_identical_model_output(self) -> None:
        response = {"question_text": "How would you validate a Python data pipeline before deployment?"}
        configure_structured_client(FakeModel([response, response]))

        first = self.generator.generate(plan(), self.competency, self.job_evidence, self.candidate_evidence)
        second = self.generator.generate(plan(), self.competency, self.job_evidence, self.candidate_evidence)

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
