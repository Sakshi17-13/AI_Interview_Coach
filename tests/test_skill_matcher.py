"""Offline tests for evidence-constrained semantic skill matching."""

from __future__ import annotations

import json
import unittest

from pydantic import ValidationError

from ai.llm_client import configure_structured_client
from ai.schemas import CandidateProfile, Evidence, RoleProfile, Skill, SkillMatch, SkillMatchReport
from ai.skill_matcher import SkillMatchGroundingError, SkillMatcher


class FakeModel:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self._responses = iter(responses)

    def chat(self, messages: list[dict[str, str]]) -> dict[str, object]:
        return {"choices": [{"message": {"content": json.dumps(next(self._responses))}}]}


def evidence(excerpt: str) -> dict[str, str]:
    return {"excerpt": excerpt}


def competency(
    competency_id: str, name: str, requirement: str, weight: float = 1.0
) -> dict[str, object]:
    return {
        "id": competency_id,
        "name": name,
        "requirements": [requirement],
        "weight": weight,
        "evidence": [evidence(requirement)],
    }


def match_response(
    competency_id: str,
    competency_name: str,
    level: str,
    weight: float,
    status: str,
    job_excerpt: str,
    candidate_excerpts: list[str] | None = None,
) -> dict[str, object]:
    return {
        "competency_id": competency_id,
        "competency_name": competency_name,
        "requirement_level": level,
        "competency_weight": weight,
        "match_status": status,
        "candidate_evidence": [evidence(item) for item in candidate_excerpts or []],
        "job_requirement_evidence": [evidence(job_excerpt)],
        "reasoning": "Offline fake semantic assessment.",
    }


class SkillMatcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.matcher = SkillMatcher()

    def test_strong_semantic_match(self) -> None:
        pipeline_evidence = "Developed machine learning pipelines using Python."
        candidate = CandidateProfile(skills=[Skill(name="Python", evidence=[Evidence(excerpt=pipeline_evidence)])])
        requirement = "Python programming"
        role = RoleProfile(required_competencies=[competency("python", "Python programming", requirement)])
        configure_structured_client(
            FakeModel([match_response("python", "Python programming", "required", 1.0, "strong", requirement, [pipeline_evidence])])
        )

        report = self.matcher.match(candidate, role)

        self.assertEqual(report.required_competency_results[0].match_status, "strong")
        self.assertEqual(report.overall_match_score, 100.0)

    def test_partial_match(self) -> None:
        sql_evidence = "Wrote SQL queries to analyze product data."
        candidate = CandidateProfile(skills=[Skill(name="SQL", evidence=[Evidence(excerpt=sql_evidence)])])
        requirement = "Experience with SQL and data warehouse platforms"
        role = RoleProfile(required_competencies=[competency("warehouse", "Data warehouse platforms", requirement)])
        configure_structured_client(
            FakeModel([match_response("warehouse", "Data warehouse platforms", "required", 1.0, "partial", requirement, [sql_evidence])])
        )

        report = self.matcher.match(candidate, role)

        self.assertEqual(report.partial_match_competency_ids, ["warehouse"])
        self.assertEqual(report.required_match_score, 50.0)

    def test_missing_docker_is_not_demonstrated(self) -> None:
        candidate = CandidateProfile(skills=[Skill(name="Python", evidence=[Evidence(excerpt="Built Python services.")])])
        requirement = "Experience with Docker containerization"
        role = RoleProfile(required_competencies=[competency("docker", "Docker/containerization", requirement)])
        configure_structured_client(
            FakeModel([match_response("docker", "Docker/containerization", "required", 1.0, "not_demonstrated", requirement)])
        )

        report = self.matcher.match(candidate, role)

        result = report.required_competency_results[0]
        self.assertEqual(result.match_status, "not_demonstrated")
        self.assertEqual(result.candidate_evidence, [])

    def test_fabricated_candidate_evidence_is_rejected(self) -> None:
        candidate = CandidateProfile(skills=[Skill(name="Python", evidence=[Evidence(excerpt="Built Python services.")])])
        requirement = "Python programming"
        role = RoleProfile(required_competencies=[competency("python", "Python programming", requirement)])
        configure_structured_client(
            FakeModel([match_response("python", "Python programming", "required", 1.0, "strong", requirement, ["Led Kubernetes at Example Corp"])])
        )

        with self.assertRaises(SkillMatchGroundingError):
            self.matcher.match(candidate, role)

    def test_unsupported_job_evidence_is_rejected(self) -> None:
        candidate_evidence = "Built Python services."
        candidate = CandidateProfile(skills=[Skill(name="Python", evidence=[Evidence(excerpt=candidate_evidence)])])
        requirement = "Python programming"
        role = RoleProfile(required_competencies=[competency("python", "Python programming", requirement)])
        configure_structured_client(
            FakeModel([match_response("python", "Python programming", "required", 1.0, "strong", "Unrelated job evidence", [candidate_evidence])])
        )

        with self.assertRaises(SkillMatchGroundingError):
            self.matcher.match(candidate, role)

    def test_required_and_preferred_results_stay_separate(self) -> None:
        candidate_evidence = "Developed machine learning pipelines using Python."
        candidate = CandidateProfile(skills=[Skill(name="Python", evidence=[Evidence(excerpt=candidate_evidence)])])
        python_requirement = "Python programming"
        docker_requirement = "Experience with Docker containerization"
        role = RoleProfile(
            required_competencies=[competency("python", "Python programming", python_requirement)],
            preferred_competencies=[competency("docker", "Docker/containerization", docker_requirement)],
        )
        configure_structured_client(
            FakeModel([
                match_response("python", "Python programming", "required", 1.0, "strong", python_requirement, [candidate_evidence]),
                match_response("docker", "Docker/containerization", "preferred", 1.0, "not_demonstrated", docker_requirement),
            ])
        )

        report = self.matcher.match(candidate, role)

        self.assertEqual([item.competency_id for item in report.required_competency_results], ["python"])
        self.assertEqual([item.competency_id for item in report.preferred_competency_results], ["docker"])

    def test_weighted_required_score_has_greater_influence_than_preferred_score(self) -> None:
        candidate_evidence = "Built Python services."
        candidate = CandidateProfile(skills=[Skill(name="Python", evidence=[Evidence(excerpt=candidate_evidence)])])
        required_requirement = "Experience with Docker containerization"
        preferred_requirement = "Python programming"
        role = RoleProfile(
            required_competencies=[competency("docker", "Docker/containerization", required_requirement, weight=5.0)],
            preferred_competencies=[competency("python", "Python programming", preferred_requirement, weight=1.0)],
        )
        configure_structured_client(
            FakeModel([
                match_response("docker", "Docker/containerization", "required", 5.0, "not_demonstrated", required_requirement),
                match_response("python", "Python programming", "preferred", 1.0, "strong", preferred_requirement, [candidate_evidence]),
            ])
        )

        report = self.matcher.match(candidate, role)

        self.assertEqual(report.required_match_score, 0.0)
        self.assertEqual(report.preferred_match_score, 100.0)
        self.assertEqual(report.overall_match_score, 20.0)

    def test_empty_candidate_profile_returns_valid_report(self) -> None:
        requirement = "Python programming"
        role = RoleProfile(required_competencies=[competency("python", "Python programming", requirement)])
        configure_structured_client(
            FakeModel([match_response("python", "Python programming", "required", 1.0, "not_demonstrated", requirement)])
        )

        report = self.matcher.match(CandidateProfile(), role)

        self.assertEqual(report.not_demonstrated_competency_ids, ["python"])

    def test_empty_role_competencies_returns_safe_report(self) -> None:
        report = self.matcher.match(CandidateProfile(), RoleProfile())

        self.assertEqual(report.overall_match_score, 0.0)
        self.assertIsNone(report.required_match_score)
        self.assertIsNone(report.preferred_match_score)

    def test_strong_without_candidate_evidence_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            SkillMatch.model_validate(match_response("python", "Python", "required", 1.0, "strong", "Python"))

    def test_partial_without_candidate_evidence_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            SkillMatch.model_validate(match_response("python", "Python", "required", 1.0, "partial", "Python"))

    def test_duplicate_competency_id_is_rejected(self) -> None:
        required = SkillMatch.model_validate(
            match_response("python", "Python", "required", 1.0, "not_demonstrated", "Python")
        )
        preferred = SkillMatch.model_validate(
            match_response("python", "Python", "preferred", 1.0, "not_demonstrated", "Python")
        )
        with self.assertRaises(ValidationError):
            SkillMatchReport(
                required_competency_results=[required],
                preferred_competency_results=[preferred],
                not_demonstrated_competency_ids=["python", "python"],
            )


if __name__ == "__main__":
    unittest.main()
