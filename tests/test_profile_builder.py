"""Offline tests for evidence-grounded CandidateProfile and RoleProfile building."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import PyPDF2
from pydantic import ValidationError

from ai.llm_client import configure_structured_client
from ai.profile_builder import ProfileGroundingError, build_candidate_profile, build_role_profile
from ai.schemas import RoleProfile


ROOT = Path(__file__).resolve().parents[1]


class FakeModel:
    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)

    def chat(self, messages: list[dict[str, str]]) -> dict[str, object]:
        return {"choices": [{"message": {"content": next(self._responses)}}]}


class ProfileBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        reader = PyPDF2.PdfReader(ROOT / "example-resume.pdf")
        cls.resume_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        cls.job_description = (ROOT / "example-job-description.txt").read_text(encoding="utf-8")

    def test_builders_accept_valid_evidence_grounded_output_from_example_inputs(self) -> None:
        candidate = {
            "candidate_name": "Jane Doe",
            "candidate_name_evidence": [{"excerpt": "Jane Doe"}],
            "headline": "Data Scientist",
            "headline_evidence": [{"excerpt": "Data Scientist"}],
            "years_experience_estimate": 5.0,
            "years_experience_evidence": [{"excerpt": "5+ year"}],
            "skills": [{"name": "Python", "evidence": [{"excerpt": "Python"}]}],
            "technologies_tools": [
                {"name": "PyTorch", "category": "library", "evidence": [{"excerpt": "PyTorch"}]}
            ],
            "experiences": [],
            "projects": [
                {
                    "id": "project_1",
                    "name": "Customer Segmentation Model",
                    "description": "Applied K-means clustering and PCA.",
                    "technologies": ["Python"],
                    "achievements": [
                        {
                            "description": "20% improvement in marketing campaign ROI.",
                            "context": "Customer Segmentation Model",
                            "evidence": [{"excerpt": "20% improvement"}],
                        }
                    ],
                    "evidence": [{"excerpt": "Customer Segmentation Model"}],
                }
            ],
            "achievements": [],
            "certifications": [],
            "education": [],
        }
        role = {
            "job_title": "Data Scientist",
            "seniority": None,
            "location": "Hybrid – New York, NY (3 days onsite per week)",
            "employment_type": "Full-time",
            "required_competencies": [
                {
                    "id": "predictive_modeling",
                    "name": "Predictive modeling",
                    "requirements": ["Develop predictive models using Python or R."],
                    "weight": 1.0,
                    "evidence": [{"excerpt": "Develop, validate, and deploy predictive"}],
                }
            ],
            "preferred_competencies": [
                {
                    "id": "cloud_containers",
                    "name": "Cloud platforms and containerization",
                    "requirements": ["Experience with cloud platforms and containerization."],
                    "weight": 1.0,
                    "evidence": [{"excerpt": "cloud platforms (AWS, GCP, Azure) and containerization"}],
                }
            ],
            "technologies_tools": [
                {
                    "name": "Python",
                    "requirement_level": "required",
                    "evidence": [{"excerpt": "Proficiency in Python"}],
                },
                {
                    "name": "Docker",
                    "requirement_level": "preferred",
                    "evidence": [{"excerpt": "containerization (Docker)"}],
                },
            ],
            "responsibilities": [
                {
                    "description": "Develop predictive and prescriptive models.",
                    "evidence": [{"excerpt": "Develop, validate, and deploy predictive"}],
                }
            ],
            "interview_focus": [
                {
                    "topic": "Predictive modeling",
                    "competency_ids": ["predictive_modeling"],
                    "evidence": [{"excerpt": "predictive and prescriptive models"}],
                }
            ],
        }
        configure_structured_client(FakeModel([f"```json\n{json.dumps(candidate)}\n```", json.dumps(role)]))

        built_candidate = build_candidate_profile(self.resume_text)
        built_role = build_role_profile(self.job_description)

        self.assertEqual(built_candidate.projects[0].name, "Customer Segmentation Model")
        self.assertEqual(
            [competency.id for competency in built_role.required_competencies],
            ["predictive_modeling"],
        )
        self.assertEqual(
            [competency.id for competency in built_role.preferred_competencies],
            ["cloud_containers"],
        )

    def test_missing_information_is_safe(self) -> None:
        configure_structured_client(FakeModel(["{}", "{}"]))

        candidate = build_candidate_profile(self.resume_text)
        role = build_role_profile(self.job_description)

        self.assertIsNone(candidate.candidate_name)
        self.assertEqual(candidate.projects, [])
        self.assertEqual(role.required_competencies, [])
        self.assertEqual(role.preferred_competencies, [])

    def test_unsupported_candidate_claim_is_rejected_when_evidence_is_not_in_resume(self) -> None:
        unsupported_candidate = {
            "skills": [
                {
                    "name": "Kubernetes leadership",
                    "evidence": [{"excerpt": "Led Kubernetes platform at Example Corp"}],
                }
            ]
        }
        configure_structured_client(FakeModel([json.dumps(unsupported_candidate)]))

        with self.assertRaises(ProfileGroundingError):
            build_candidate_profile(self.resume_text)

    def test_required_and_preferred_competencies_cannot_overlap(self) -> None:
        duplicate_competency = {
            "id": "python",
            "name": "Python",
            "requirements": ["Python"],
            "weight": 1.0,
            "evidence": [{"excerpt": "Python"}],
        }
        with self.assertRaises(ValidationError):
            RoleProfile.model_validate(
                {
                    "required_competencies": [duplicate_competency],
                    "preferred_competencies": [duplicate_competency],
                }
            )


if __name__ == "__main__":
    unittest.main()
