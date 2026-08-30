"""Offline hardening tests for the application event controller."""

from __future__ import annotations

import unittest

from application_controller import (
    end_application_interview,
    start_application_interview,
    submit_application_answer,
)


class Question:
    def __init__(self, text: str) -> None:
        self.question_text = text


class Engine:
    def __init__(self, complete: bool = False) -> None:
        self.complete = complete

    def is_complete(self) -> bool:
        return self.complete


class Session:
    def __init__(self) -> None:
        self.engine = Engine()
        self.ended_early = False
        self.needs_question_retry = False
        self.transcripts: list[str] = []
        self.raise_submit: Exception | None = None
        self.report: object = "final report"
        self.report_error: Exception | None = None
        self.recovery_question = Question("Recovered question?")
        self.end_calls = 0

    def submit_transcript(self, transcript: str):
        self.transcripts.append(transcript)
        if self.raise_submit:
            raise self.raise_submit
        return type("Turn", (), {"report": None, "next_question": Question("Next question?")})()

    def retry_next_question(self):
        self.needs_question_retry = False
        return self.recovery_question

    def build_final_report(self):
        if self.report_error:
            raise self.report_error
        return self.report

    def end_early(self):
        self.ended_early = True
        self.end_calls += 1
        return self.build_final_report()


class ApplicationControllerTests(unittest.TestCase):
    def start(self, **overrides):
        calls: list[str] = []
        session = overrides.pop("session", Session())
        callbacks = {
            "extract_pdf": lambda path: calls.append("extract") or "resume text",
            "build_candidate": lambda text: calls.append("candidate") or "candidate",
            "build_role": lambda text: calls.append("role") or "role",
            "create_session": lambda candidate, role, count: (calls.append("session") or (session, Question("First question?"))),
            "text_to_speech": lambda text: calls.append(f"tts:{text}") or f"audio-{len(calls)}.mp3",
        }
        callbacks.update(overrides)
        return start_application_interview("resume.pdf", "job description", 3, **callbacks), calls, session

    def test_valid_resume_setup_keeps_question_text_and_audio_path_separate(self) -> None:
        update, calls, session = self.start()
        self.assertEqual(update.question_text, "First question?")
        self.assertNotEqual(update.question_text, update.question_audio_path)
        self.assertEqual(calls, ["extract", "candidate", "role", "session", "tts:First question?"])
        self.assertIs(update.session, session)

    def test_pdf_empty_resume_job_and_ai_setup_failures_reset_session(self) -> None:
        cases = [
            {"extract_pdf": lambda _: (_ for _ in ()).throw(ValueError("corrupted PDF"))},
            {"extract_pdf": lambda _: "   "},
            {"build_candidate": lambda _: (_ for _ in ()).throw(ValueError("invalid candidate JSON"))},
            {"build_role": lambda _: (_ for _ in ()).throw(ValueError("invalid role JSON"))},
            {"create_session": lambda *_: (_ for _ in ()).throw(RuntimeError("matching/init failed"))},
        ]
        for callbacks in cases:
            update, _, _ = self.start(**callbacks)
            self.assertIsNone(update.session)
            self.assertIn("❌ Unable to start interview", update.status)
        update = start_application_interview(
            "resume.pdf", " ", 3,
            extract_pdf=lambda _: "resume", build_candidate=lambda _: "candidate",
            build_role=lambda _: "role", create_session=lambda *_: (Session(), Question("never")), text_to_speech=lambda _: "audio",
        )
        self.assertIsNone(update.session)
        self.assertIn("job description", update.status)

    def test_empty_recording_blank_transcript_and_transcription_error(self) -> None:
        session = Session()
        blank = submit_application_answer(session, None, transcribe=lambda _: "", text_to_speech=lambda _: "audio", format_report=str)
        self.assertEqual(session.transcripts, [""])
        self.assertEqual(blank.question_text, "Next question?")
        failed = submit_application_answer(session, "audio.wav", transcribe=lambda _: "❌ Whisper failed", text_to_speech=lambda _: "audio", format_report=str)
        self.assertEqual(failed.status, "❌ Whisper failed")
        self.assertEqual(len(session.transcripts), 1)

    def test_assessment_failure_keeps_session_and_allows_retry(self) -> None:
        session = Session()
        session.raise_submit = RuntimeError("invalid assessment JSON")
        failed = submit_application_answer(session, "audio", transcribe=lambda _: "exact transcript", text_to_speech=lambda _: "audio", format_report=str)
        self.assertIs(failed.session, session)
        self.assertEqual(failed.submit_label, "Submit Answer")
        session.raise_submit = None
        retry = submit_application_answer(session, "audio", transcribe=lambda _: "exact transcript", text_to_speech=lambda _: "audio", format_report=str)
        self.assertEqual(session.transcripts, ["exact transcript", "exact transcript"])
        self.assertEqual(retry.question_text, "Next question?")

    def test_question_generation_failure_uses_question_only_retry(self) -> None:
        session = Session()
        session.needs_question_retry = True
        update = submit_application_answer(session, "ignored.wav", transcribe=lambda _: self.fail("STT should not run"), text_to_speech=lambda text: f"audio:{text}", format_report=str)
        self.assertEqual(update.question_text, "Recovered question?")
        self.assertEqual(update.question_audio_path, "audio:Recovered question?")
        self.assertEqual(session.transcripts, [])

    def test_final_report_failure_is_retryable_without_another_answer(self) -> None:
        session = Session()
        session.engine.complete = True
        session.report_error = RuntimeError("malformed report response")
        failed = submit_application_answer(session, "ignored", transcribe=lambda _: self.fail("STT should not run"), text_to_speech=lambda _: "audio", format_report=str)
        self.assertEqual(failed.submit_label, "Retry Final Report")
        self.assertEqual(session.transcripts, [])
        session.report_error = None
        retry = submit_application_answer(session, "ignored", transcribe=lambda _: self.fail("STT should not run"), text_to_speech=lambda _: "audio", format_report=lambda value: f"REPORT:{value}")
        self.assertEqual(retry.status, "REPORT:final report")

    def test_report_failure_after_final_answer_keeps_completed_session_for_retry(self) -> None:
        session = Session()

        def complete_then_fail(_: str):
            session.engine.complete = True
            raise RuntimeError("report model failed")

        session.submit_transcript = complete_then_fail
        failed = submit_application_answer(session, "audio", transcribe=lambda _: "answer", text_to_speech=lambda _: "audio", format_report=str)
        self.assertEqual(failed.submit_label, "Retry Final Report")
        self.assertIs(failed.session, session)

    def test_early_termination_is_session_local(self) -> None:
        first, second = Session(), Session()
        update = end_application_interview(first, format_report=lambda value: f"INCOMPLETE:{value}")
        self.assertEqual(update.status, "INCOMPLETE:final report")
        self.assertTrue(first.ended_early)
        self.assertFalse(second.ended_early)
        blocked = submit_application_answer(first, "audio", transcribe=lambda _: "answer", text_to_speech=lambda _: "audio", format_report=str)
        self.assertEqual(blocked.status, "final report")
        self.assertEqual(first.transcripts, [])

    def test_tts_failures_do_not_replace_the_generated_question(self) -> None:
        session = Session()
        update = submit_application_answer(session, "audio", transcribe=lambda _: "answer", text_to_speech=lambda _: (_ for _ in ()).throw(RuntimeError("gTTS failed")), format_report=str)
        self.assertEqual(update.question_text, "Next question?")
        self.assertIsNone(update.question_audio_path)
        self.assertIn("audio rendering failed", update.status)


if __name__ == "__main__":
    unittest.main()
