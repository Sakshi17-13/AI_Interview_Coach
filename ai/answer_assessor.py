"""Evidence-grounded assessment of a single candidate answer."""

from __future__ import annotations

from .llm_client import StructuredOutputError, get_structured_client
from .prompts import answer_assessment_messages
from .schemas import (
    AnswerAssessment,
    AnswerAssessmentContent,
    AssessmentDimensions,
    Competency,
    Evidence,
    InterviewQuestion,
    QuestionPlan,
)


class AnswerAssessmentError(ValueError):
    """Raised when answer assessment data is unsafe or inconsistent with its input."""


_FORBIDDEN_INTERNAL_TERMS = (
    "competency id",
    "competency_id",
    "planner",
    "system instruction",
    "internal score",
    "selection logic",
)


def normalize_transcript_whitespace(text: str) -> str:
    """Normalize whitespace for blank-answer detection only, never for evidence matching."""

    return " ".join(text.split())


class AnswerAssessor:
    """Assesses one transcript and never changes planning or question metadata."""

    def assess(
        self,
        interview_question: InterviewQuestion,
        answer_transcript: str | None,
        selected_competency: Competency | None = None,
        job_requirement_evidence: list[Evidence] | None = None,
        candidate_evidence: list[Evidence] | None = None,
        question_plan: QuestionPlan | None = None,
    ) -> AnswerAssessment:
        transcript = answer_transcript or ""
        job_evidence = job_requirement_evidence or []
        candidate_context = candidate_evidence or []
        self._validate_context(
            interview_question,
            selected_competency,
            job_evidence,
            question_plan,
        )
        if not normalize_transcript_whitespace(transcript):
            return self._empty_assessment(interview_question)

        question_payload = {
            "question_text": interview_question.question_text,
            "question_type": interview_question.question_type,
            "objective": interview_question.objective,
            "expected_evidence": interview_question.expected_evidence,
        }
        competency_payload = (
            {"name": selected_competency.name} if selected_competency is not None else None
        )
        try:
            content = get_structured_client().generate_structured(
                answer_assessment_messages(
                    question_payload,
                    competency_payload,
                    [item.model_dump() for item in job_evidence],
                    [item.model_dump() for item in candidate_context],
                    transcript,
                    AnswerAssessmentContent.model_json_schema(),
                ),
                AnswerAssessmentContent,
            )
        except StructuredOutputError as error:
            raise AnswerAssessmentError("The model did not return a valid answer assessment.") from error

        self._validate_answer_evidence(content, transcript)
        self._validate_assessment_language(content)
        return AnswerAssessment(
            question_id=interview_question.question_id,
            competency_id=interview_question.competency_id,
            overall_score=content.overall_score,
            dimensions=AssessmentDimensions(
                technical_accuracy=content.technical_accuracy,
                relevance=content.relevance,
                specificity=content.specificity,
                communication=content.communication,
            ),
            evidence=content.evidence,
            strengths=content.strengths,
            gaps=content.gaps,
            recommended_action=content.recommended_action,
            follow_up_focus=content.follow_up_focus,
            confidence=content.confidence,
        )

    @staticmethod
    def _validate_context(
        question: InterviewQuestion,
        competency: Competency | None,
        job_evidence: list[Evidence],
        plan: QuestionPlan | None,
    ) -> None:
        if question.competency_id is None:
            if competency is not None or job_evidence:
                raise AnswerAssessmentError("General questions cannot use competency-specific context.")
        else:
            if competency is None or competency.id != question.competency_id:
                raise AnswerAssessmentError("Selected competency must match InterviewQuestion.competency_id.")
            allowed_job = {item.excerpt for item in competency.evidence}
            if any(item.excerpt not in allowed_job for item in job_evidence):
                raise AnswerAssessmentError("Job evidence must belong to the selected competency.")
        if plan is not None:
            comparisons = (
                (question.question_id, plan.question_id),
                (question.competency_id, plan.competency_id),
                (question.question_type, plan.question_type),
                (question.difficulty, plan.difficulty),
                (question.objective, plan.objective),
                (question.expected_evidence, plan.expected_evidence),
                (question.follow_up_of, plan.follow_up_of),
            )
            if any(left != right for left, right in comparisons):
                raise AnswerAssessmentError("QuestionPlan must exactly match the InterviewQuestion metadata.")

    @staticmethod
    def _empty_assessment(question: InterviewQuestion) -> AnswerAssessment:
        return AnswerAssessment(
            question_id=question.question_id,
            competency_id=question.competency_id,
            overall_score=0,
            dimensions=AssessmentDimensions(
                technical_accuracy=0, relevance=0, specificity=0, communication=0
            ),
            evidence=[],
            strengths=[],
            gaps=["No substantive answer was provided."],
            recommended_action="change_topic",
            follow_up_focus=None,
            confidence=1.0,
        )

    @staticmethod
    def _validate_answer_evidence(content: AnswerAssessmentContent, transcript: str) -> None:
        for evidence in content.evidence:
            if evidence.source != "answer":
                raise AnswerAssessmentError("Assessment evidence must use source='answer'.")
            if evidence.excerpt not in transcript:
                raise AnswerAssessmentError(
                    "Assessment cites evidence that is not verbatim in the answer transcript: "
                    f"{evidence.excerpt!r}"
                )

    @staticmethod
    def _validate_assessment_language(content: AnswerAssessmentContent) -> None:
        text = " ".join([*content.strengths, *content.gaps, content.follow_up_focus or ""])
        lowered = text.casefold()
        if any(term in lowered for term in _FORBIDDEN_INTERNAL_TERMS):
            raise AnswerAssessmentError("Assessment exposes internal planning or scoring instructions.")
