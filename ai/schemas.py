"""Validated contracts for information extracted from application inputs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SchemaModel(BaseModel):
    """Base model that rejects fields the application has not defined."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Evidence(SchemaModel):
    """A short excerpt that lets later stages trace a claim to its source input."""

    excerpt: str = Field(min_length=1, description="Verbatim or near-verbatim source excerpt.")


class Skill(SchemaModel):
    name: str = Field(min_length=1, description="Skill explicitly named in the resume.")
    evidence: list[Evidence] = Field(min_length=1)


class Technology(SchemaModel):
    """A named tool, platform, framework, library, or language from a resume."""

    name: str = Field(min_length=1)
    category: str | None = Field(
        default=None,
        description="Use only a clearly supported category such as language, cloud, or tool.",
    )
    evidence: list[Evidence] = Field(min_length=1)


class Achievement(SchemaModel):
    description: str = Field(min_length=1)
    context: str | None = None
    evidence: list[Evidence] = Field(min_length=1)


class Experience(SchemaModel):
    id: str = Field(min_length=1, description="Stable identifier such as exp_1.")
    organization: str | None = None
    role: str | None = None
    duration: str | None = Field(default=None, description="Use only stated dates/duration.")
    highlights: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    achievements: list[Achievement] = Field(default_factory=list)
    evidence: list[Evidence] = Field(min_length=1)


class Project(SchemaModel):
    id: str = Field(min_length=1, description="Stable identifier such as project_1.")
    name: str = Field(min_length=1)
    description: str | None = None
    technologies: list[str] = Field(default_factory=list)
    achievements: list[Achievement] = Field(default_factory=list)
    evidence: list[Evidence] = Field(min_length=1)


class Certification(SchemaModel):
    name: str = Field(min_length=1)
    issuer: str | None = None
    date: str | None = None
    credential_id: str | None = None
    evidence: list[Evidence] = Field(min_length=1)


class Education(SchemaModel):
    degree: str | None = None
    field: str | None = None
    institution: str | None = None
    graduation_date: str | None = None
    evidence: list[Evidence] = Field(min_length=1)


class CandidateProfile(SchemaModel):
    candidate_name: str | None = None
    candidate_name_evidence: list[Evidence] = Field(default_factory=list)
    headline: str | None = Field(
        default=None, description="Resume-supported professional summary or current title."
    )
    headline_evidence: list[Evidence] = Field(default_factory=list)
    years_experience_estimate: float | None = Field(default=None, ge=0)
    years_experience_evidence: list[Evidence] = Field(default_factory=list)
    skills: list[Skill] = Field(default_factory=list)
    technologies_tools: list[Technology] = Field(default_factory=list)
    experiences: list[Experience] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    achievements: list[Achievement] = Field(default_factory=list)
    certifications: list[Certification] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_evidence_for_profile_facts(self) -> "CandidateProfile":
        fact_evidence = (
            ("candidate_name", self.candidate_name, self.candidate_name_evidence),
            ("headline", self.headline, self.headline_evidence),
            (
                "years_experience_estimate",
                self.years_experience_estimate,
                self.years_experience_evidence,
            ),
        )
        for field_name, value, evidence in fact_evidence:
            if value is not None and not evidence:
                raise ValueError(f"{field_name} requires supporting resume evidence.")
        return self


class Competency(SchemaModel):
    id: str = Field(min_length=1, description="Stable identifier such as python_data_work.")
    name: str = Field(min_length=1)
    requirements: list[str] = Field(default_factory=list)
    weight: float = Field(
        default=1.0,
        gt=0,
        le=5.0,
        description="Relative importance, grounded only in explicit job-description priority signals.",
    )
    evidence: list[Evidence] = Field(min_length=1)


class TechnologyRequirement(SchemaModel):
    name: str = Field(min_length=1)
    requirement_level: Literal["required", "preferred"]
    evidence: list[Evidence] = Field(min_length=1)


class Responsibility(SchemaModel):
    description: str = Field(min_length=1)
    evidence: list[Evidence] = Field(min_length=1)


class InterviewFocus(SchemaModel):
    topic: str = Field(min_length=1)
    competency_ids: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(min_length=1)


class RoleProfile(SchemaModel):
    job_title: str | None = None
    seniority: str | None = None
    location: str | None = None
    employment_type: str | None = None
    required_competencies: list[Competency] = Field(default_factory=list)
    preferred_competencies: list[Competency] = Field(default_factory=list)
    technologies_tools: list[TechnologyRequirement] = Field(default_factory=list)
    responsibilities: list[Responsibility] = Field(default_factory=list)
    interview_focus: list[InterviewFocus] = Field(default_factory=list)

    @model_validator(mode="after")
    def keep_competency_levels_distinct(self) -> "RoleProfile":
        required_ids = {competency.id for competency in self.required_competencies}
        preferred_ids = {competency.id for competency in self.preferred_competencies}
        duplicates = required_ids.intersection(preferred_ids)
        if duplicates:
            raise ValueError(
                "A competency cannot be both required and preferred: "
                f"{', '.join(sorted(duplicates))}."
            )
        return self


class SkillMatch(SchemaModel):
    """Evidence-backed comparison of one candidate profile to one role competency."""

    competency_id: str = Field(min_length=1)
    competency_name: str = Field(min_length=1)
    requirement_level: Literal["required", "preferred"]
    competency_weight: float = Field(gt=0, le=5.0)
    match_status: Literal["strong", "partial", "not_demonstrated"]
    candidate_evidence: list[Evidence] = Field(default_factory=list)
    job_requirement_evidence: list[Evidence] = Field(min_length=1)
    reasoning: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_evidence_for_positive_matches(self) -> "SkillMatch":
        if self.match_status in {"strong", "partial"} and not self.candidate_evidence:
            raise ValueError("strong and partial matches require candidate evidence.")
        if self.match_status == "not_demonstrated" and self.candidate_evidence:
            raise ValueError(
                "not_demonstrated means not evidenced in the provided profile and must not "
                "contain candidate evidence."
            )
        return self


class MatchArea(SchemaModel):
    competency_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    rationale: str = Field(min_length=1)


class SkillMatchReport(SchemaModel):
    overall_match_score: float = Field(default=0.0, ge=0, le=100)
    required_match_score: float | None = Field(default=None, ge=0, le=100)
    preferred_match_score: float | None = Field(default=None, ge=0, le=100)
    required_competency_results: list[SkillMatch] = Field(default_factory=list)
    preferred_competency_results: list[SkillMatch] = Field(default_factory=list)
    matched_competency_ids: list[str] = Field(default_factory=list)
    partial_match_competency_ids: list[str] = Field(default_factory=list)
    not_demonstrated_competency_ids: list[str] = Field(default_factory=list)
    strongest_areas: list[MatchArea] = Field(default_factory=list)
    weakest_areas: list[MatchArea] = Field(default_factory=list)
    recommended_interview_focus: list[MatchArea] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_result_membership_and_areas(self) -> "SkillMatchReport":
        required_ids = [match.competency_id for match in self.required_competency_results]
        preferred_ids = [match.competency_id for match in self.preferred_competency_results]
        all_ids = required_ids + preferred_ids
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("Every competency ID must occur only once in a SkillMatchReport.")
        if any(match.requirement_level != "required" for match in self.required_competency_results):
            raise ValueError("Required results must contain only required competencies.")
        if any(match.requirement_level != "preferred" for match in self.preferred_competency_results):
            raise ValueError("Preferred results must contain only preferred competencies.")

        status_ids = {
            "strong": [match.competency_id for match in self.required_competency_results + self.preferred_competency_results if match.match_status == "strong"],
            "partial": [match.competency_id for match in self.required_competency_results + self.preferred_competency_results if match.match_status == "partial"],
            "not_demonstrated": [match.competency_id for match in self.required_competency_results + self.preferred_competency_results if match.match_status == "not_demonstrated"],
        }
        if self.matched_competency_ids != status_ids["strong"]:
            raise ValueError("matched_competency_ids must contain exactly the strong-match IDs.")
        if self.partial_match_competency_ids != status_ids["partial"]:
            raise ValueError("partial_match_competency_ids must contain exactly the partial-match IDs.")
        if self.not_demonstrated_competency_ids != status_ids["not_demonstrated"]:
            raise ValueError(
                "not_demonstrated_competency_ids must contain exactly the not-demonstrated IDs."
            )

        area_ids = {
            area.competency_id
            for area in self.strongest_areas + self.weakest_areas + self.recommended_interview_focus
        }
        unknown_area_ids = area_ids.difference(all_ids)
        if unknown_area_ids:
            raise ValueError(
                "MatchArea competency IDs must refer to report results: "
                f"{', '.join(sorted(unknown_area_ids))}."
            )
        return self


class QuestionGuidance(SchemaModel):
    """LLM-generated content for a deterministically selected interview question."""

    objective: str = Field(min_length=1)
    expected_evidence: list[str] = Field(default_factory=list)


class QuestionPlan(SchemaModel):
    question_id: str = Field(min_length=1)
    competency_id: str | None = None
    stage: Literal["introduction", "resume", "technical", "behavioral", "scenario", "closing"]
    question_type: Literal[
        "open_ended",
        "technical",
        "behavioral",
        "project_deep_dive",
        "scenario",
        "follow_up",
    ]
    difficulty: int = Field(ge=1, le=5)
    objective: str = Field(min_length=1)
    expected_evidence: list[str] = Field(default_factory=list)
    follow_up_of: str | None = None
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_follow_up_shape(self) -> "QuestionPlan":
        if self.question_type == "follow_up" and not self.follow_up_of:
            raise ValueError("follow_up questions require follow_up_of.")
        if self.question_type != "follow_up" and self.follow_up_of is not None:
            raise ValueError("Only follow_up questions may set follow_up_of.")
        return self


class QuestionText(SchemaModel):
    """The only field the question-generation LLM is permitted to supply."""

    question_text: str = Field(min_length=1)


class InterviewQuestion(SchemaModel):
    """A rendered question whose strategy metadata is copied from a QuestionPlan."""

    question_id: str = Field(min_length=1)
    question_text: str = Field(min_length=1)
    competency_id: str | None = None
    question_type: Literal[
        "open_ended",
        "technical",
        "behavioral",
        "project_deep_dive",
        "scenario",
        "follow_up",
    ]
    difficulty: int = Field(ge=1, le=5)
    objective: str = Field(min_length=1)
    expected_evidence: list[str] = Field(default_factory=list)
    follow_up_of: str | None = None

    @model_validator(mode="after")
    def validate_follow_up_shape(self) -> "InterviewQuestion":
        if self.question_type == "follow_up" and not self.follow_up_of:
            raise ValueError("follow_up questions require follow_up_of.")
        if self.question_type != "follow_up" and self.follow_up_of is not None:
            raise ValueError("Only follow_up questions may set follow_up_of.")
        return self


class AnswerEvidence(SchemaModel):
    """Verbatim support taken solely from the current answer transcript."""

    excerpt: str = Field(min_length=1)
    source: Literal["answer"] = "answer"


class AssessmentDimensions(SchemaModel):
    technical_accuracy: float = Field(ge=0, le=5)
    relevance: float = Field(ge=0, le=5)
    specificity: float = Field(ge=0, le=5)
    communication: float = Field(ge=0, le=5)


class AnswerAssessmentContent(SchemaModel):
    """The assessment fields that the LLM may produce for one answer."""

    overall_score: float = Field(ge=0, le=5)
    technical_accuracy: float = Field(ge=0, le=5)
    relevance: float = Field(ge=0, le=5)
    specificity: float = Field(ge=0, le=5)
    communication: float = Field(ge=0, le=5)
    evidence: list[AnswerEvidence] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    recommended_action: Literal["follow_up", "advance", "change_topic"]
    follow_up_focus: str | None = None
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_follow_up_focus(self) -> "AnswerAssessmentContent":
        if self.recommended_action == "follow_up" and not self.follow_up_focus:
            raise ValueError("follow_up requires follow_up_focus.")
        if self.recommended_action != "follow_up" and self.follow_up_focus is not None:
            raise ValueError("advance and change_topic require follow_up_focus to be null.")
        return self


class AnswerAssessment(SchemaModel):
    """Evidence-grounded evaluation of one transcript against one interview question."""

    question_id: str = Field(min_length=1)
    competency_id: str | None = None
    overall_score: float = Field(ge=0, le=5)
    dimensions: AssessmentDimensions
    evidence: list[AnswerEvidence] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    recommended_action: Literal["follow_up", "advance", "change_topic"]
    follow_up_focus: str | None = None
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_follow_up_focus(self) -> "AnswerAssessment":
        if self.recommended_action == "follow_up" and not self.follow_up_focus:
            raise ValueError("follow_up requires follow_up_focus.")
        if self.recommended_action != "follow_up" and self.follow_up_focus is not None:
            raise ValueError("advance and change_topic require follow_up_focus to be null.")
        return self


class InterviewCoverage(SchemaModel):
    competency_id: str = Field(min_length=1)
    questions_asked: int = Field(default=0, ge=0)
    last_score: float | None = Field(default=None, ge=0, le=5)
    coverage_status: Literal["not_covered", "partial", "covered"] = "not_covered"


class PreviousAnswerSignal(SchemaModel):
    """A narrow planner input supplied by a later answer-assessment phase."""

    question_id: str = Field(min_length=1)
    competency_id: str = Field(min_length=1)
    score: float = Field(ge=0, le=5)
    gaps: list[str] = Field(default_factory=list)


class InterviewPlanState(SchemaModel):
    target_question_count: int = Field(ge=1)
    questions_generated: int = Field(default=0, ge=0)
    current_stage: Literal["introduction", "resume", "technical", "behavioral", "scenario", "closing"] = "introduction"
    coverage: list[InterviewCoverage] = Field(default_factory=list)
    follow_up_count: int = Field(default=0, ge=0)
    max_follow_ups: int = Field(default=2, ge=0)
    question_history: list[QuestionPlan] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_history_and_limits(self) -> "InterviewPlanState":
        if self.questions_generated != len(self.question_history):
            raise ValueError("questions_generated must equal the number of planned questions.")
        if self.questions_generated > self.target_question_count:
            raise ValueError("Question count cannot exceed target_question_count.")
        coverage_ids = [item.competency_id for item in self.coverage]
        if len(coverage_ids) != len(set(coverage_ids)):
            raise ValueError("Coverage may contain each competency ID only once.")

        question_ids = [question.question_id for question in self.question_history]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("Question IDs must be unique.")
        seen_question_ids: set[str] = set()
        calculated_follow_ups = 0
        for question in self.question_history:
            if question.follow_up_of is not None and question.follow_up_of not in seen_question_ids:
                raise ValueError("follow_up_of must reference an earlier question.")
            if question.question_type == "follow_up":
                calculated_follow_ups += 1
            seen_question_ids.add(question.question_id)
        if self.follow_up_count != calculated_follow_ups:
            raise ValueError("follow_up_count must equal planned follow-up questions.")
        if self.follow_up_count > self.max_follow_ups:
            raise ValueError("follow_up_count cannot exceed max_follow_ups.")
        return self
