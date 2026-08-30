"""Deterministic final interview reporting with bounded qualitative assistance."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from .llm_client import StructuredOutputError, get_structured_client
from .prompts import final_report_qualitative_messages
from .schemas import (
    AnswerAssessment,
    Competency,
    CompetencyResult,
    DimensionResult,
    FinalInterviewReport,
    InterviewSessionState,
    InterviewStatistics,
    QualitativeReportContent,
    QualitativeFinding,
    ReportEvidence,
)


class ReportBuilderError(ValueError):
    """Raised when a report cannot be safely assembled from interview state."""


_DIMENSION_WEIGHTS = {
    "technical_accuracy": 0.35,
    "relevance": 0.25,
    "specificity": 0.20,
    "communication": 0.20,
}
_FORBIDDEN_QUALITATIVE_TERMS = (
    "competency id",
    "competency_id",
    "weight",
    "score",
    "planner",
    "system instruction",
    "hidden reasoning",
    "resume says",
)
_GENERIC_CAPITALIZED = {
    "A", "An", "And", "Candidate", "Clear", "Focus", "For", "Improve", "In", "Next",
    "Provide", "The", "They", "This", "To", "Use", "Work",
}


class FinalReportBuilder:
    """Build a report without allowing the LLM to determine metrics or job fit.

    Overall score is the weighted mean of all assessment dimensions: technical
    accuracy 35%, relevance 25%, specificity 20%, and communication 20%.
    Job fit is the weighted competency score, using 80% required and 20%
    preferred performance when both levels were assessed.
    """

    def build(self, interview_state: InterviewSessionState) -> FinalInterviewReport:
        """Assemble an evidence-grounded completed or explicitly incomplete report."""

        self._validate_state(interview_state)
        assessments_by_competency = self._assessments_by_competency(interview_state)
        evidence = self._report_evidence(interview_state)
        dimension_results = self._dimension_results(interview_state, evidence)
        overall_score = self._overall_score(dimension_results)
        competency_results = self._competency_results(interview_state, assessments_by_competency)
        job_fit_score, required_score = self._job_fit_score(competency_results)
        statistics = self._statistics(interview_state, competency_results)
        qualitative = self._qualitative_content(
            interview_state,
            competency_results,
            dimension_results,
            evidence,
            statistics,
        )
        return FinalInterviewReport(
            status="completed" if interview_state.status == "completed" else "incomplete",
            overall_score=overall_score,
            job_fit_score=job_fit_score,
            recommendation=self._recommendation(job_fit_score, overall_score, required_score),
            competency_results=competency_results,
            dimension_results=dimension_results,
            strongest_areas=self._finding_texts(qualitative.strongest_areas),
            weakest_areas=self._finding_texts(qualitative.weakest_areas),
            communication_feedback=self._finding_texts(qualitative.communication_feedback),
            technical_feedback=self._finding_texts(qualitative.technical_feedback),
            improvement_recommendations=self._finding_texts(qualitative.improvement_recommendations),
            next_steps=self._finding_texts(qualitative.next_steps),
            interview_statistics=statistics,
        )

    def build_from_engine(self, interview_engine: Any) -> FinalInterviewReport:
        """Convenience API that consumes the engine's immutable state snapshot."""

        return self.build(interview_engine.get_state())

    @staticmethod
    def _validate_state(state: InterviewSessionState) -> None:
        question_ids = {question.question_id for question in state.generated_questions}
        answer_ids = [record.question_id for record in state.answer_history]
        if len(answer_ids) != len(set(answer_ids)):
            raise ReportBuilderError("A final report requires at most one assessment per question.")
        if any(question_id not in question_ids for question_id in answer_ids):
            raise ReportBuilderError("Every answer record must belong to a generated interview question.")

    @staticmethod
    def _assessments_by_competency(
        state: InterviewSessionState,
    ) -> dict[str, list[AnswerAssessment]]:
        grouped: dict[str, list[AnswerAssessment]] = defaultdict(list)
        for record in state.answer_history:
            if record.assessment.competency_id is not None:
                grouped[record.assessment.competency_id].append(record.assessment)
        return dict(grouped)

    @staticmethod
    def _report_evidence(state: InterviewSessionState) -> list[ReportEvidence]:
        evidence: list[ReportEvidence] = []
        for record in state.answer_history:
            if record.assessment.question_id != record.question_id:
                raise ReportBuilderError("Assessment question ID must match its answer record.")
            for item in record.assessment.evidence:
                if item.source != "answer" or item.excerpt not in record.transcript:
                    raise ReportBuilderError(
                        "Report evidence must be verbatim answer evidence from its original transcript."
                    )
                evidence.append(
                    ReportEvidence(
                        question_id=record.question_id, excerpt=item.excerpt, source=item.source
                    )
                )
        return evidence

    def _dimension_results(
        self, state: InterviewSessionState, evidence: list[ReportEvidence]
    ) -> list[DimensionResult]:
        assessments = [record.assessment for record in state.answer_history]
        strengths = self._unique(item for assessment in assessments for item in assessment.strengths)
        weaknesses = self._unique(item for assessment in assessments for item in assessment.gaps)
        results: list[DimensionResult] = []
        for dimension in _DIMENSION_WEIGHTS:
            average = (
                sum(getattr(item.dimensions, dimension) for item in assessments) / len(assessments)
                if assessments
                else 0.0
            )
            results.append(
                DimensionResult(
                    dimension=dimension,
                    score=round(average, 2),
                    evidence=evidence,
                    strengths=strengths,
                    weaknesses=weaknesses,
                )
            )
        return results

    def _competency_results(
        self,
        state: InterviewSessionState,
        grouped: dict[str, list[AnswerAssessment]],
    ) -> list[CompetencyResult]:
        role_competencies = {
            item.id: (item, "required") for item in state.role_profile.required_competencies
        }
        role_competencies.update(
            {item.id: (item, "preferred") for item in state.role_profile.preferred_competencies}
        )
        coverage = {item.competency_id: item for item in state.plan_state.coverage}
        questions = {item.question_id: item for item in state.generated_questions}
        results: list[CompetencyResult] = []
        for competency_id, assessments in grouped.items():
            item = role_competencies.get(competency_id)
            if item is None:
                raise ReportBuilderError("An assessment references a competency absent from the role profile.")
            competency, level = item
            assessment_evidence = [
                ReportEvidence(question_id=assessment.question_id, excerpt=evidence.excerpt, source=evidence.source)
                for assessment in assessments
                for evidence in assessment.evidence
            ]
            follow_up_count = sum(
                1
                for assessment in assessments
                if questions.get(assessment.question_id) is not None
                and questions[assessment.question_id].question_type == "follow_up"
            )
            coverage_item = coverage.get(competency_id)
            results.append(
                CompetencyResult(
                    competency_id=competency.id,
                    competency_name=competency.name,
                    requirement_level=level,
                    competency_weight=competency.weight,
                    interview_score=round(
                        sum(item.overall_score for item in assessments) / len(assessments) / 5 * 100,
                        1,
                    ),
                    question_count=len(assessments),
                    follow_up_count=follow_up_count,
                    coverage_status=(
                        coverage_item.coverage_status if coverage_item is not None else "not_covered"
                    ),
                    strengths=self._unique(value for item in assessments for value in item.strengths),
                    weaknesses=self._unique(value for item in assessments for value in item.gaps),
                    evidence=assessment_evidence,
                    improvement_actions=self._unique(value for item in assessments for value in item.gaps),
                )
            )
        return sorted(results, key=lambda item: (item.requirement_level != "required", item.competency_id))

    @staticmethod
    def _overall_score(dimension_results: list[DimensionResult]) -> float:
        values = {item.dimension: item.score for item in dimension_results}
        return round(sum(values[dimension] / 5 * 100 * weight for dimension, weight in _DIMENSION_WEIGHTS.items()), 1)

    @staticmethod
    def _weighted_level_score(results: list[CompetencyResult]) -> float | None:
        if not results:
            return None
        denominator = sum(item.competency_weight for item in results)
        return round(sum(item.interview_score * item.competency_weight for item in results) / denominator, 1)

    def _job_fit_score(self, results: list[CompetencyResult]) -> tuple[float, float | None]:
        required = self._weighted_level_score(
            [item for item in results if item.requirement_level == "required"]
        )
        preferred = self._weighted_level_score(
            [item for item in results if item.requirement_level == "preferred"]
        )
        if required is not None and preferred is not None:
            return round(required * 0.8 + preferred * 0.2, 1), required
        if required is not None:
            return required, required
        if preferred is not None:
            return preferred, None
        return 0.0, None

    @staticmethod
    def _recommendation(job_fit_score: float, overall_score: float, required_score: float | None) -> str:
        # Required performance that is materially below overall performance caps the recommendation.
        capped_score = job_fit_score
        if required_score is not None and required_score < max(60.0, overall_score - 15.0):
            capped_score = min(capped_score, 59.9 if required_score < 60 else 74.9)
        if capped_score >= 90:
            return "strong_match"
        if capped_score >= 75:
            return "good_match"
        if capped_score >= 60:
            return "developing_match"
        return "weak_match"

    @staticmethod
    def _statistics(
        state: InterviewSessionState, competency_results: list[CompetencyResult]
    ) -> InterviewStatistics:
        answered_ids = {record.question_id for record in state.answer_history}
        follow_up_count = sum(
            question.question_type == "follow_up" for question in state.generated_questions
        )
        return InterviewStatistics(
            total_questions=len(state.generated_questions),
            answered_questions=len(answered_ids),
            follow_up_questions=follow_up_count,
            competencies_assessed=len(competency_results),
            competencies_sufficiently_covered=sum(
                item.coverage_status == "covered" for item in competency_results
            ),
        )

    def _qualitative_content(
        self,
        state: InterviewSessionState,
        competencies: list[CompetencyResult],
        dimensions: list[DimensionResult],
        evidence: list[ReportEvidence],
        statistics: InterviewStatistics,
    ) -> QualitativeReportContent:
        # With no validated answer excerpts, qualitative prose would have no permissible
        # evidence base. Keep the report deterministically empty instead of calling the LLM.
        if not state.answer_history or not evidence:
            return QualitativeReportContent()
        role_context = {
            "job_title": state.role_profile.job_title,
            "required_competencies": [self._competency_context(item) for item in state.role_profile.required_competencies],
            "preferred_competencies": [self._competency_context(item) for item in state.role_profile.preferred_competencies],
        }
        skill_match_context = {
            "required_match_statuses": [
                {"name": item.competency_name, "status": item.match_status}
                for item in state.skill_match_report.required_competency_results
            ],
            "preferred_match_statuses": [
                {"name": item.competency_name, "status": item.match_status}
                for item in state.skill_match_report.preferred_competency_results
            ],
        }
        try:
            content = get_structured_client().generate_structured(
                final_report_qualitative_messages(
                    role_context,
                    [item.model_dump() for item in competencies],
                    [item.model_dump() for item in dimensions],
                    [item.model_dump() for item in evidence],
                    statistics.model_dump(),
                    skill_match_context,
                    QualitativeReportContent.model_json_schema(),
                ),
                QualitativeReportContent,
            )
        except StructuredOutputError as error:
            raise ReportBuilderError("The model did not return valid qualitative report content.") from error
        self._validate_qualitative_content(content, competencies, evidence)
        return content

    @staticmethod
    def _competency_context(competency: Competency) -> dict[str, Any]:
        return {
            "name": competency.name,
            "requirements": competency.requirements,
            "evidence": [item.model_dump() for item in competency.evidence],
        }

    def _validate_qualitative_content(
        self,
        content: QualitativeReportContent,
        competency_results: list[CompetencyResult],
        evidence: list[ReportEvidence],
    ) -> None:
        findings = [
            *content.strongest_areas,
            *content.weakest_areas,
            *content.communication_feedback,
            *content.technical_feedback,
            *content.improvement_recommendations,
            *content.next_steps,
        ]
        allowed_evidence = {(item.question_id, item.excerpt, item.source) for item in evidence}
        for finding in findings:
            if any(
                (item.question_id, item.excerpt, item.source) not in allowed_evidence
                for item in finding.evidence
            ):
                raise ReportBuilderError(
                    "Qualitative feedback must reference only validated assessment evidence."
                )
        text = " ".join(item.text for item in findings)
        lowered = text.casefold()
        if any(term in lowered for term in _FORBIDDEN_QUALITATIVE_TERMS):
            raise ReportBuilderError("Qualitative feedback exposes prohibited internal report content.")
        allowed_context = " ".join(
            [
                *(item.excerpt for item in evidence),
                *(value for item in competency_results for value in item.strengths),
                *(value for item in competency_results for value in item.weaknesses),
                *(item.competency_name for item in competency_results),
            ]
        )
        allowed_named = set(re.findall(r"\b[A-Z][A-Za-z0-9+#.-]*\b", allowed_context))
        for token in re.findall(r"\b[A-Z][A-Za-z0-9+#.-]*\b", text):
            if token not in allowed_named and token not in _GENERIC_CAPITALIZED:
                raise ReportBuilderError(f"Qualitative feedback introduces an unsupported named detail: {token!r}")
        for result in competency_results:
            if (
                re.search(r"\bcandidate\b", lowered)
                and result.competency_name.casefold() in lowered
                and not any(result.competency_name.casefold() in item.excerpt.casefold() for item in evidence)
            ):
                raise ReportBuilderError(
                    "Qualitative feedback treats profile or job context as interview-answer evidence."
                )

    @staticmethod
    def _unique(values: Any) -> list[str]:
        return list(dict.fromkeys(values))

    @staticmethod
    def _finding_texts(findings: list[QualitativeFinding]) -> list[str]:
        return [item.text for item in findings]
