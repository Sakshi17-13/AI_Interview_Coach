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
