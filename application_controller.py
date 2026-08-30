"""Dependency-injected UI controller for robust, session-local interview events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from interview_app_adapter import InterviewApplicationSession


@dataclass(frozen=True)
class ApplicationUpdate:
    question_text: str
    question_audio_path: str | None
    status: str
    submit_label: str
    session: InterviewApplicationSession | None


def start_application_interview(
    resume_path: str | object,
    job_description: str,
    target_question_count: int,
    *,
    extract_pdf: Callable[[str | object], str],
    build_candidate: Callable[[str], Any],
    build_role: Callable[[str], Any],
    create_session: Callable[[Any, Any, int], tuple[InterviewApplicationSession, Any]],
    text_to_speech: Callable[[str], str],
) -> ApplicationUpdate:
    """Start fresh state only after every setup step succeeds."""

    try:
        resume_text = extract_pdf(resume_path)
        if not resume_text or not resume_text.strip():
            raise ValueError("The resume PDF does not contain extractable text.")
        if not job_description or not job_description.strip():
            raise ValueError("A job description is required.")
        candidate_profile = build_candidate(resume_text)
        role_profile = build_role(job_description)
        session, question = create_session(candidate_profile, role_profile, int(target_question_count))
        audio_path = text_to_speech(question.question_text)
        return ApplicationUpdate(
            question.question_text,
            audio_path,
            "Interview in progress. Submit a recorded answer for each question.",
            "Submit Answer",
            session,
        )
    except Exception as error:
        return ApplicationUpdate("", None, f"❌ Unable to start interview: {error}", "Start Interview", None)


def submit_application_answer(
    session: InterviewApplicationSession | None,
    answer_audio_path: str | None,
    *,
    transcribe: Callable[[str | None], str],
    text_to_speech: Callable[[str], str],
    format_report: Callable[[Any], str],
) -> ApplicationUpdate:
    """Advance, recover a pending question, or retry a final report without state reuse."""

    if session is None:
        return ApplicationUpdate("", None, "❌ Start an interview before submitting an answer.", "Start Interview", None)
    if session.engine.is_complete() or session.ended_early:
        try:
            return ApplicationUpdate("", None, format_report(session.build_final_report()), "Interview Complete", session)
        except Exception as error:
            return ApplicationUpdate("", None, f"❌ Final report is unavailable; retry: {error}", "Retry Final Report", session)
    if session.needs_question_retry:
        try:
            question = session.retry_next_question()
            if question is None:
                return ApplicationUpdate("", None, "Interview completed without a final report.", "Interview Complete", session)
            return ApplicationUpdate(question.question_text, text_to_speech(question.question_text), "Next question recovered. Submit an answer when ready.", "Submit Answer", session)
        except Exception as error:
            return ApplicationUpdate("", None, f"❌ Unable to generate next question; retry: {error}", "Retry Next Question", session)

    transcript = transcribe(answer_audio_path)
    if transcript.startswith("❌"):
        return ApplicationUpdate("", None, transcript, "Submit Answer", session)
    try:
        turn = session.submit_transcript(transcript)
    except Exception as error:
        if session.engine.is_complete():
            return ApplicationUpdate("", None, f"❌ Final report is unavailable; retry: {error}", "Retry Final Report", session)
        label = "Retry Next Question" if session.needs_question_retry else "Submit Answer"
        return ApplicationUpdate("", None, f"❌ Unable to process answer: {error}", label, session)
    if turn.report is not None:
        return ApplicationUpdate("", None, format_report(turn.report), "Interview Complete", session)
    if turn.next_question is None:
        return ApplicationUpdate("", None, "Interview completed without a final report.", "Interview Complete", session)
    try:
        audio_path = text_to_speech(turn.next_question.question_text)
    except Exception as error:
        return ApplicationUpdate(turn.next_question.question_text, None, f"❌ Next question generated, but audio rendering failed: {error}", "Submit Answer", session)
    return ApplicationUpdate(turn.next_question.question_text, audio_path, "Interview in progress. Submit a recorded answer for the next question.", "Submit Answer", session)


def end_application_interview(
    session: InterviewApplicationSession | None,
    *,
    format_report: Callable[[Any], str],
) -> ApplicationUpdate:
    """End one session without inventing completion; backend report is marked incomplete."""

    if session is None:
        return ApplicationUpdate("", None, "❌ Start an interview before ending it.", "Start Interview", None)
    try:
        return ApplicationUpdate("", None, format_report(session.end_early()), "Interview Complete", session)
    except Exception as error:
        return ApplicationUpdate("", None, f"❌ Unable to generate incomplete report; retry: {error}", "Retry Final Report", session)
