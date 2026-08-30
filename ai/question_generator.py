"""Render planner-owned question intent into one validated interview question."""

from __future__ import annotations

import re

from .llm_client import StructuredOutputError, get_structured_client
from .prompts import interview_question_messages
from .schemas import (
    Competency,
    Evidence,
    InterviewQuestion,
    PreviousAnswerSignal,
    QuestionPlan,
    QuestionText,
)


class QuestionGenerationError(ValueError):
    """Raised when generated wording is empty, unsafe, or inconsistent with its context."""


_FORBIDDEN_INTERNAL_PHRASES = (
    "competency id",
    "competency_id",
    "planner",
    "system instruction",
    "expected evidence",
    "question plan",
    "because your resume says",
    "your resume says",
    "internal score",
)
_GENERIC_CAPITALIZED_WORDS = {
    "Can", "Could", "Describe", "Explain", "How", "In", "Please", "Share", "Tell", "What",
    "When", "Where", "Why", "Would", "You", "Your",
}


def normalize_question_whitespace(text: str) -> str:
    """Collapse formatting whitespace while preserving wording and punctuation."""

    return " ".join(text.split())


class QuestionGenerator:
    """Generates wording only; all interview strategy remains owned by QuestionPlan."""

    def generate(
        self,
        question_plan: QuestionPlan,
        selected_competency: Competency | None = None,
        job_requirement_evidence: list[Evidence] | None = None,
        candidate_evidence: list[Evidence] | None = None,
        previous_question: str | None = None,
        previous_answer_signal: PreviousAnswerSignal | None = None,
    ) -> InterviewQuestion:
        candidate_evidence = candidate_evidence or []
        job_requirement_evidence = job_requirement_evidence or []
        self._validate_context(
            question_plan,
            selected_competency,
            job_requirement_evidence,
            candidate_evidence,
            previous_question,
            previous_answer_signal,
        )
        plan_payload = {
            "objective": question_plan.objective,
            "question_type": question_plan.question_type,
            "difficulty": question_plan.difficulty,
            "expected_evidence": question_plan.expected_evidence,
        }
        competency_payload = (
            {"name": selected_competency.name} if selected_competency is not None else None
        )
        prior_signal_payload = (
            {"gaps": previous_answer_signal.gaps} if previous_answer_signal is not None else None
        )
        try:
            generated = get_structured_client().generate_structured(
                interview_question_messages(
                    plan_payload,
                    competency_payload,
                    [evidence.model_dump() for evidence in job_requirement_evidence],
                    [evidence.model_dump() for evidence in candidate_evidence],
                    previous_question,
                    prior_signal_payload,
                    QuestionText.model_json_schema(),
                ),
                QuestionText,
            )
        except StructuredOutputError as error:
            raise QuestionGenerationError("The model did not return a valid interview question.") from error
        question_text = normalize_question_whitespace(generated.question_text)
        self._validate_question_text(question_text, question_plan, selected_competency, job_requirement_evidence, candidate_evidence)
        return InterviewQuestion(
            question_id=question_plan.question_id,
            question_text=question_text,
            competency_id=question_plan.competency_id,
            question_type=question_plan.question_type,
            difficulty=question_plan.difficulty,
            objective=question_plan.objective,
            expected_evidence=question_plan.expected_evidence,
            follow_up_of=question_plan.follow_up_of,
        )

    @staticmethod
    def _validate_context(
        plan: QuestionPlan,
        competency: Competency | None,
        job_evidence: list[Evidence],
        candidate_evidence: list[Evidence],
        previous_question: str | None,
        previous_signal: PreviousAnswerSignal | None,
    ) -> None:
        if plan.competency_id is None and competency is not None:
            raise QuestionGenerationError("A general plan cannot be paired with a competency.")
        if plan.competency_id is not None:
            if competency is None or competency.id != plan.competency_id:
                raise QuestionGenerationError("Selected competency must match QuestionPlan.competency_id.")
            permitted_job_evidence = {QuestionGenerator._normalise(item.excerpt) for item in competency.evidence}
            if not job_evidence:
                raise QuestionGenerationError("Competency questions require job requirement evidence.")
            if any(QuestionGenerator._normalise(item.excerpt) not in permitted_job_evidence for item in job_evidence):
                raise QuestionGenerationError("Job requirement evidence must belong to the selected competency.")
        elif job_evidence or candidate_evidence:
            raise QuestionGenerationError("General questions may not receive competency-specific evidence.")
        if plan.question_type == "project_deep_dive" and not candidate_evidence:
            raise QuestionGenerationError("Project deep dives require supplied candidate evidence.")
        if plan.question_type == "follow_up":
            if not previous_question:
                raise QuestionGenerationError("Follow-up questions require the previous question text.")
            if previous_signal is not None and previous_signal.competency_id != plan.competency_id:
                raise QuestionGenerationError("Previous-answer signal must match the planned competency.")

    @staticmethod
    def _normalise(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", text.casefold())

    def _validate_question_text(
        self,
        question_text: str,
        plan: QuestionPlan,
        competency: Competency | None,
        job_evidence: list[Evidence],
        candidate_evidence: list[Evidence],
    ) -> None:
        if not question_text:
            raise QuestionGenerationError("Generated question text cannot be empty.")
        if question_text.count("?") != 1 or not question_text.endswith("?"):
            raise QuestionGenerationError("Generated output must contain exactly one question ending in '?'.")
        lowered = question_text.casefold()
        if any(phrase in lowered for phrase in _FORBIDDEN_INTERNAL_PHRASES):
            raise QuestionGenerationError("Generated question exposes internal interview metadata.")
        self._reject_unknown_named_details(
            question_text,
            plan,
            competency,
            job_evidence,
            candidate_evidence,
        )

    def _reject_unknown_named_details(
        self,
        question_text: str,
        plan: QuestionPlan,
        competency: Competency | None,
        job_evidence: list[Evidence],
        candidate_evidence: list[Evidence],
    ) -> None:
        allowed_context = " ".join(
            [
                plan.objective,
                *plan.expected_evidence,
                competency.name if competency is not None else "",
                *(item.excerpt for item in job_evidence),
                *(item.excerpt for item in candidate_evidence),
            ]
        )
        allowed_named = set(re.findall(r"\b[A-Z][A-Za-z0-9+#.-]*\b", allowed_context))
        named_tokens = re.findall(r"\b[A-Z][A-Za-z0-9+#.-]*\b", question_text)
        for token in named_tokens[1:]:
            if token not in allowed_named and token not in _GENERIC_CAPITALIZED_WORDS:
                raise QuestionGenerationError(
                    f"Generated question introduces an unsupported named detail: {token!r}"
                )
