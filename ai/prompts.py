"""Prompts used by the structured profile-building layer."""

from __future__ import annotations

import json
from typing import Any


def _schema_text(schema: dict[str, Any]) -> str:
    return json.dumps(schema, indent=2)


def resume_analysis_messages(resume_text: str, schema: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You extract a candidate profile from a resume. Return only a JSON object "
                "that satisfies the supplied JSON Schema. Use only facts explicitly supported "
                "by the resume. Do not infer demographic information, skills, job titles, dates, "
                "institutions, or experience. Use null or [] when information is absent. "
                "For years_experience_estimate, use a value only when stated dates make it clear. "
                "Extract projects, achievements, certifications, and technologies/tools only when "
                "they are explicitly present. For every evidence field, include a short verbatim "
                "excerpt from the resume that supports the associated claim. When candidate_name, "
                "headline, or years_experience_estimate is present, also populate its corresponding "
                "*_evidence field. Do not "
                "turn an unsupported inference into a project, achievement, or certification."
            ),
        },
        {
            "role": "user",
            "content": f"JSON Schema:\n{_schema_text(schema)}\n\nResume:\n{resume_text}",
        },
    ]


def job_description_analysis_messages(
    job_description: str, schema: dict[str, Any]
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You extract a role profile from a job description. Return only a JSON object "
                "that satisfies the supplied JSON Schema. Keep explicitly required items in "
                "required_competencies and explicitly preferred, desired, or nice-to-have items "
                "in preferred_competencies. Responsibilities must be job duties, not copied "
                "qualifications. Extract technologies/tools and label each required or preferred "
                "from the wording in the job description. interview_focus should be concise interview "
                "areas grounded in stated competencies or responsibilities, with the relevant competency "
                "IDs when applicable. Include a short verbatim job-description excerpt "
                "in every evidence field. Assign a competency weight above the default only when the "
                "job description explicitly signals greater importance; otherwise use 1.0. Do not invent "
                "requirements, priorities, or evidence."
            ),
        },
        {
            "role": "user",
            "content": (
                f"JSON Schema:\n{_schema_text(schema)}\n\n"
                f"Job description:\n{job_description}"
            ),
        },
    ]


def competency_match_messages(
    candidate_evidence_inventory: list[dict[str, str]],
    competency: dict[str, Any],
    schema: dict[str, Any],
) -> list[dict[str, str]]:
    """Create a bounded semantic-matching request for one role competency."""

    return [
        {
            "role": "system",
            "content": (
                "You compare exactly one job competency to candidate evidence. Return only a JSON "
                "object that satisfies the supplied JSON Schema. Reason semantically, not by exact "
                "keyword overlap, but use ONLY the supplied candidate evidence and ONLY the supplied "
                "job requirement evidence. Never invent skills, experience, technologies, projects, "
                "or achievements. Return strong only when the evidence directly and substantially "
                "supports the competency; return partial when it directly supports a meaningful but "
                "incomplete portion; otherwise return not_demonstrated. not_demonstrated means the "
                "provided profile does not demonstrate the competency, not that the candidate lacks it. "
                "For strong or partial, candidate_evidence must copy one or more supplied candidate "
                "evidence excerpts exactly. For not_demonstrated, candidate_evidence must be empty. "
                "job_requirement_evidence must copy supplied job evidence excerpts exactly. Keep "
                "reasoning concise. Do not treat related experience as evidence of an unstated skill: "
                "Python pipeline experience can support Python programming, but cannot support Docker "
                "without Docker/container evidence."
            ),
        },
        {
            "role": "user",
            "content": (
                f"JSON Schema:\n{_schema_text(schema)}\n\n"
                f"Competency:\n{json.dumps(competency, indent=2)}\n\n"
                f"Candidate evidence inventory:\n{json.dumps(candidate_evidence_inventory, indent=2)}"
            ),
        },
    ]


def question_guidance_messages(
    selected_competency: dict[str, Any] | None,
    candidate_evidence: list[dict[str, str]],
    coverage: dict[str, Any] | None,
    previous_answer_signal: dict[str, Any] | None,
    schema: dict[str, Any],
) -> list[dict[str, str]]:
    """Request question content after deterministic planning has selected the target."""

    return [
        {
            "role": "system",
            "content": (
                "Generate only the objective and expected evidence for one already-selected interview "
                "question. Return only a JSON object that satisfies the supplied JSON Schema. Use ONLY "
                "the selected competency, supplied job requirement evidence, supplied candidate evidence, "
                "coverage, and previous-answer signal. Never invent candidate experience, skills, projects, "
                "technologies, job requirements, or previous answers. When a previous-answer gap is supplied, "
                "focus the objective on that stated gap. Keep the objective concise and make expected_evidence "
                "observable interview evidence rather than claims about the candidate."
            ),
        },
        {
            "role": "user",
            "content": (
                f"JSON Schema:\n{_schema_text(schema)}\n\n"
                f"Selected competency:\n{json.dumps(selected_competency, indent=2)}\n\n"
                f"Candidate evidence:\n{json.dumps(candidate_evidence, indent=2)}\n\n"
                f"Interview coverage:\n{json.dumps(coverage, indent=2)}\n\n"
                f"Previous answer signal:\n{json.dumps(previous_answer_signal, indent=2)}"
            ),
        },
    ]


def interview_question_messages(
    question_plan: dict[str, Any],
    competency: dict[str, Any] | None,
    job_requirement_evidence: list[dict[str, str]],
    candidate_evidence: list[dict[str, str]],
    previous_question: str | None,
    previous_answer_signal: dict[str, Any] | None,
    schema: dict[str, Any],
) -> list[dict[str, str]]:
    """Ask the LLM only for wording after the planner has fixed the interview strategy."""

    return [
        {
            "role": "system",
            "content": (
                "Write exactly one natural, professional interview question. Return only JSON matching "
                "the supplied schema. The only output field is question_text. Follow the supplied question "
                "plan exactly: do not change its objective, question type, difficulty, competency, stage, or "
                "follow-up relationship. Use ONLY the supplied candidate evidence, job requirement evidence, "
                "and optional previous-question context. Never invent candidate experience, skills, projects, "
                "technologies, job requirements, or prior answers. Do not mention competency IDs, scores, "
                "planner decisions, system instructions, or why the question was selected. Ask one question, "
                "ending in a single question mark. For project deep dives, reference only supplied candidate "
                "evidence. For scenarios, ground the scenario in the job evidence. For follow-ups, probe only "
                "the supplied prior-answer gap."
            ),
        },
        {
            "role": "user",
            "content": (
                f"JSON Schema:\n{_schema_text(schema)}\n\n"
                f"Question plan:\n{json.dumps(question_plan, indent=2)}\n\n"
                f"Selected competency:\n{json.dumps(competency, indent=2)}\n\n"
                f"Job requirement evidence:\n{json.dumps(job_requirement_evidence, indent=2)}\n\n"
                f"Candidate evidence:\n{json.dumps(candidate_evidence, indent=2)}\n\n"
                f"Previous question:\n{json.dumps(previous_question)}\n\n"
                f"Previous answer signal:\n{json.dumps(previous_answer_signal, indent=2)}"
            ),
        },
    ]


def answer_assessment_messages(
    interview_question: dict[str, Any],
    competency: dict[str, Any] | None,
    job_requirement_evidence: list[dict[str, str]],
    candidate_evidence: list[dict[str, str]],
    answer_transcript: str,
    schema: dict[str, Any],
) -> list[dict[str, str]]:
    """Assess one answer while treating its exact transcript as the only answer-evidence source."""

    return [
        {
            "role": "system",
            "content": (
                "Assess exactly one candidate answer against one interview question. Return only JSON "
                "matching the supplied schema. The candidate answer transcript is the sole source of "
                "answer evidence: every evidence excerpt must be copied verbatim from it and source must "
                "be 'answer'. Resume/profile evidence is context for relevance only and cannot prove the "
                "candidate said or demonstrated anything in this answer. Never invent answer content, "
                "experience, skills, projects, technologies, job requirements, or personality traits. "
                "Score 0–5 for technical accuracy, relevance, specificity, communication, and overall "
                "quality; do not blindly average dimensions if the answer is irrelevant or incorrect. "
                "Recommend follow_up only for a specific unresolved gap and provide follow_up_focus then; "
                "otherwise use advance or change_topic with null follow_up_focus. Confidence is confidence "
                "in this assessment, not candidate confidence. Do not mention planner instructions, internal "
                "scores, competency IDs, or selection logic."
            ),
        },
        {
            "role": "user",
            "content": (
                f"JSON Schema:\n{_schema_text(schema)}\n\n"
                f"Interview question:\n{json.dumps(interview_question, indent=2)}\n\n"
                f"Selected competency:\n{json.dumps(competency, indent=2)}\n\n"
                f"Job requirement evidence:\n{json.dumps(job_requirement_evidence, indent=2)}\n\n"
                f"Candidate profile evidence (context only):\n{json.dumps(candidate_evidence, indent=2)}\n\n"
                f"Candidate answer transcript:\n{answer_transcript}"
            ),
        },
    ]


def json_repair_messages(
    invalid_output: str, schema: dict[str, Any], error: str
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Repair the supplied model output. Return only one valid JSON object that "
                "conforms exactly to the supplied JSON Schema. Do not add commentary or Markdown."
            ),
        },
        {
            "role": "user",
            "content": (
                f"JSON Schema:\n{_schema_text(schema)}\n\n"
                f"Validation error:\n{error}\n\n"
                f"Invalid output:\n{invalid_output}"
            ),
        },
    ]
