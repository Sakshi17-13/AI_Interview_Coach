"""Offline application-adapter tests with fake completed AI components."""

from __future__ import annotations

import unittest

from interview_app_adapter import create_interview_session
from ai.schemas import (
    AnswerAssessment,
    AssessmentDimensions,
    CandidateProfile,
    Competency,
    Evidence,
    InterviewQuestion,
    RoleProfile,
    SkillMatch,
    SkillMatchReport,
)


def question(question_id: str) -> InterviewQuestion:
    return InterviewQuestion(
        question_id=question_id, question_text=f"Question {question_id}?", competency_id=None,
        question_type="open_ended", difficulty=1, objective="Introduce the candidate.",
        expected_evidence=[], follow_up_of=None,
    )


class FakeMatcher:
    def __init__(self, report: SkillMatchReport) -> None:
        self.report = report
        self.calls = 0

    def match(self, candidate, role):
        self.calls += 1
        return self.report


class FakeEngine:
    def __init__(self) -> None:
        self.transcripts: list[str] = []
        self.complete = False
        self.initialized = None

    def initialize_interview(self, candidate, role, report, target, session_id=None):
        self.initialized = (candidate, role, report, target, session_id)
        return question("q_001")

    def submit_answer(self, transcript):
        self.transcripts.append(transcript)
        self.complete = True
        assessment = AnswerAssessment(
            question_id="q_001", competency_id=None, overall_score=4,
            dimensions=AssessmentDimensions(technical_accuracy=4, relevance=4, specificity=4, communication=4),
            evidence=[], strengths=[], gaps=[], recommended_action="advance", follow_up_focus=None, confidence=1,
        )
        return type("Result", (), {"assessment": assessment, "next_question": None})()

    def is_complete(self):
        return self.complete


class FakeReportBuilder:
    def __init__(self) -> None:
        self.engine = None

    def build_from_engine(self, engine):
        self.engine = engine
        return "fake report"


class InterviewApplicationAdapterTests(unittest.TestCase):
    def test_session_builds_match_initializes_engine_and_preserves_exact_transcript(self) -> None:
        item = Competency(id="python", name="Python", requirements=["Python"], evidence=[Evidence(excerpt="Python")])
        role = RoleProfile(required_competencies=[item])
        match = SkillMatch(
            competency_id="python", competency_name="Python", requirement_level="required", competency_weight=1,
            match_status="not_demonstrated", candidate_evidence=[], job_requirement_evidence=item.evidence, reasoning="fixture",
        )
        matcher = FakeMatcher(SkillMatchReport(required_competency_results=[match], not_demonstrated_competency_ids=["python"]))
        fake_engine = FakeEngine()
        fake_report_builder = FakeReportBuilder()

        session, first = create_interview_session(
            CandidateProfile(), role, 3, skill_matcher=matcher, engine_factory=lambda: fake_engine,
            report_builder=fake_report_builder, session_id="browser-session",
        )
        turn = session.submit_transcript("  exact Whisper transcript  ")

        self.assertEqual(first.question_text, "Question q_001?")
        self.assertEqual(matcher.calls, 1)
        self.assertEqual(fake_engine.initialized[3:], (3, "browser-session"))
        self.assertEqual(fake_engine.transcripts, ["  exact Whisper transcript  "])
        self.assertEqual(turn.report, "fake report")
        self.assertIs(fake_report_builder.engine, fake_engine)

    def test_sessions_are_isolated(self) -> None:
        role = RoleProfile()
        report = SkillMatchReport()
        first_engine, second_engine = FakeEngine(), FakeEngine()
        first, _ = create_interview_session(
            CandidateProfile(), role, 2, skill_matcher=FakeMatcher(report),
            engine_factory=lambda: first_engine, report_builder=FakeReportBuilder(),
        )
        second, _ = create_interview_session(
            CandidateProfile(), role, 2, skill_matcher=FakeMatcher(report),
            engine_factory=lambda: second_engine, report_builder=FakeReportBuilder(),
        )

        first.submit_transcript("first")
        self.assertEqual(first_engine.transcripts, ["first"])
        self.assertEqual(second_engine.transcripts, [])
        self.assertIsNot(first.engine, second.engine)


if __name__ == "__main__":
    unittest.main()
