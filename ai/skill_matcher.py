"""Evidence-constrained semantic matching between candidate and role profiles."""

from __future__ import annotations

import re
from collections.abc import Iterable

from .llm_client import get_structured_client
from .prompts import competency_match_messages
from .schemas import (
    Achievement,
    CandidateProfile,
    Competency,
    Evidence,
    MatchArea,
    RoleProfile,
    SkillMatch,
    SkillMatchReport,
)


class SkillMatchGroundingError(ValueError):
    """Raised when the model cites evidence outside the supplied profiles."""


MATCH_VALUES = {"strong": 1.0, "partial": 0.5, "not_demonstrated": 0.0}


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def _append_inventory_item(
    inventory: list[dict[str, str]], seen: set[str], excerpt: str, source: str
) -> None:
    normalised = _normalise(excerpt)
    if normalised and normalised not in seen:
        seen.add(normalised)
        inventory.append({"source": source, "excerpt": excerpt})


def _add_achievement_evidence(
    inventory: list[dict[str, str]], seen: set[str], achievements: Iterable[Achievement], source: str
) -> None:
    for achievement in achievements:
        _append_inventory_item(inventory, seen, achievement.description, source)
        for evidence in achievement.evidence:
            _append_inventory_item(inventory, seen, evidence.excerpt, source)


class SkillMatcher:
    """Builds deterministic reports from independently LLM-evaluated competencies."""

    def match(self, candidate_profile: CandidateProfile, role_profile: RoleProfile) -> SkillMatchReport:
        candidate_inventory = self._candidate_evidence_inventory(candidate_profile)
        required_results = [
            self._match_competency(candidate_inventory, competency, "required")
            for competency in role_profile.required_competencies
        ]
        preferred_results = [
            self._match_competency(candidate_inventory, competency, "preferred")
            for competency in role_profile.preferred_competencies
        ]
        return self._build_report(required_results, preferred_results)

    def _match_competency(
        self,
        candidate_inventory: list[dict[str, str]],
        competency: Competency,
        requirement_level: str,
    ) -> SkillMatch:
        competency_payload = {
            "id": competency.id,
            "name": competency.name,
            "requirement_level": requirement_level,
            "weight": competency.weight,
            "requirements": competency.requirements,
            "job_requirement_evidence": [evidence.model_dump() for evidence in competency.evidence],
        }
        match = get_structured_client().generate_structured(
            competency_match_messages(
                candidate_inventory, competency_payload, SkillMatch.model_json_schema()
            ),
            SkillMatch,
        )
        self._validate_match_identity(match, competency, requirement_level)
        self._validate_match_evidence(match, candidate_inventory, competency)
        return match

    @staticmethod
    def _candidate_evidence_inventory(candidate: CandidateProfile) -> list[dict[str, str]]:
        inventory: list[dict[str, str]] = []
        seen: set[str] = set()
        for skill in candidate.skills:
            for evidence in skill.evidence:
                _append_inventory_item(inventory, seen, evidence.excerpt, f"skill:{skill.name}")
        for technology in candidate.technologies_tools:
            for evidence in technology.evidence:
                _append_inventory_item(inventory, seen, evidence.excerpt, f"technology:{technology.name}")
        for experience in candidate.experiences:
            for evidence in experience.evidence:
                _append_inventory_item(inventory, seen, evidence.excerpt, f"experience:{experience.id}")
            for highlight in experience.highlights:
                _append_inventory_item(inventory, seen, highlight, f"experience:{experience.id}")
            _add_achievement_evidence(inventory, seen, experience.achievements, f"experience:{experience.id}")
        for project in candidate.projects:
            for evidence in project.evidence:
                _append_inventory_item(inventory, seen, evidence.excerpt, f"project:{project.id}")
            if project.description:
                _append_inventory_item(inventory, seen, project.description, f"project:{project.id}")
            _add_achievement_evidence(inventory, seen, project.achievements, f"project:{project.id}")
        _add_achievement_evidence(inventory, seen, candidate.achievements, "achievement")
        for certification in candidate.certifications:
            for evidence in certification.evidence:
                _append_inventory_item(inventory, seen, evidence.excerpt, f"certification:{certification.name}")
        return inventory

    @staticmethod
    def _validate_match_identity(
        match: SkillMatch, competency: Competency, requirement_level: str
    ) -> None:
        expected = (competency.id, competency.name, requirement_level, competency.weight)
        actual = (
            match.competency_id,
            match.competency_name,
            match.requirement_level,
            match.competency_weight,
        )
        if actual != expected:
            raise SkillMatchGroundingError(
                "Model returned a match for a different competency or requirement level."
            )

    @staticmethod
    def _validate_match_evidence(
        match: SkillMatch, candidate_inventory: list[dict[str, str]], competency: Competency
    ) -> None:
        allowed_candidate_evidence = {_normalise(item["excerpt"]) for item in candidate_inventory}
        allowed_job_evidence = {_normalise(evidence.excerpt) for evidence in competency.evidence}
        for evidence in match.candidate_evidence:
            if _normalise(evidence.excerpt) not in allowed_candidate_evidence:
                raise SkillMatchGroundingError(
                    f"Match cites candidate evidence absent from CandidateProfile: {evidence.excerpt!r}"
                )
        for evidence in match.job_requirement_evidence:
            if _normalise(evidence.excerpt) not in allowed_job_evidence:
                raise SkillMatchGroundingError(
                    f"Match cites job evidence outside competency {competency.id}: {evidence.excerpt!r}"
                )

    @staticmethod
    def _score(matches: list[SkillMatch]) -> float | None:
        if not matches:
            return None
        weighted_total = sum(match.competency_weight * MATCH_VALUES[match.match_status] for match in matches)
        total_weight = sum(match.competency_weight for match in matches)
        return round(weighted_total / total_weight * 100, 1)

    def _build_report(
        self, required_results: list[SkillMatch], preferred_results: list[SkillMatch]
    ) -> SkillMatchReport:
        required_score = self._score(required_results)
        preferred_score = self._score(preferred_results)
        if required_score is not None and preferred_score is not None:
            overall_score = round(0.8 * required_score + 0.2 * preferred_score, 1)
        elif required_score is not None:
            overall_score = required_score
        elif preferred_score is not None:
            overall_score = preferred_score
        else:
            overall_score = 0.0

        all_results = required_results + preferred_results
        matched_ids = [match.competency_id for match in all_results if match.match_status == "strong"]
        partial_ids = [match.competency_id for match in all_results if match.match_status == "partial"]
        not_demonstrated_ids = [
            match.competency_id for match in all_results if match.match_status == "not_demonstrated"
        ]
        strongest = self._strongest_areas(all_results)
        weaknesses = self._weakest_results(required_results, preferred_results)
        weakest_areas = [self._area_for_weakness(match) for match in weaknesses]
        recommended_focus = [self._area_for_focus(match) for match in weaknesses]
        return SkillMatchReport(
            overall_match_score=overall_score,
            required_match_score=required_score,
            preferred_match_score=preferred_score,
            required_competency_results=required_results,
            preferred_competency_results=preferred_results,
            matched_competency_ids=matched_ids,
            partial_match_competency_ids=partial_ids,
            not_demonstrated_competency_ids=not_demonstrated_ids,
            strongest_areas=strongest,
            weakest_areas=weakest_areas,
            recommended_interview_focus=recommended_focus,
        )

    @staticmethod
    def _strongest_areas(matches: list[SkillMatch]) -> list[MatchArea]:
        strong_matches = sorted(
            (match for match in matches if match.match_status == "strong"),
            key=lambda match: match.competency_weight,
            reverse=True,
        )[:3]
        return [
            MatchArea(
                competency_id=match.competency_id,
                title=match.competency_name,
                rationale=f"Strong, profile-backed evidence for this competency (weight {match.competency_weight:g}).",
            )
            for match in strong_matches
        ]

    @staticmethod
    def _weakest_results(
        required_results: list[SkillMatch], preferred_results: list[SkillMatch]
    ) -> list[SkillMatch]:
        weaknesses = [
            match
            for match in required_results + preferred_results
            if match.match_status in {"partial", "not_demonstrated"}
        ]
        return sorted(
            weaknesses,
            key=lambda match: (
                0 if match.requirement_level == "required" else 1,
                -match.competency_weight,
            ),
        )[:3]

    @staticmethod
    def _area_for_weakness(match: SkillMatch) -> MatchArea:
        status = "not demonstrated in the provided profile" if match.match_status == "not_demonstrated" else "partially demonstrated"
        return MatchArea(
            competency_id=match.competency_id,
            title=match.competency_name,
            rationale=f"{match.requirement_level.title()} competency {status} (weight {match.competency_weight:g}).",
        )

    @staticmethod
    def _area_for_focus(match: SkillMatch) -> MatchArea:
        return MatchArea(
            competency_id=match.competency_id,
            title=match.competency_name,
            rationale=(
                f"Prioritize this {match.requirement_level} competency because the provided profile "
                f"is {match.match_status.replace('_', ' ')} for it."
            ),
        )
