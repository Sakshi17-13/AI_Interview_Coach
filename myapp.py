"""Gradio adapter for the session-local AI Interview Coach backend."""

from __future__ import annotations

import os
import tempfile

import gradio as gr
import PyPDF2
from faster_whisper import WhisperModel
from gtts import gTTS
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters

from ai import build_candidate_profile, build_role_profile, configure_structured_client
from application_controller import (
    end_application_interview,
    start_application_interview,
    submit_application_answer,
)
from interview_app_adapter import InterviewApplicationSession, create_interview_session
from watsonx_configuration import (
    CONFIGURATION_REQUIRED_MESSAGE,
    WatsonxConfiguration,
    configuration_required_ui_message,
    initialize_watsonx_from_environment,
)


def _create_watsonx_parameters() -> TextChatParameters:
    sample_params = TextChatParameters.get_sample_params()
    sample_params["max_tokens"] = int(1e5)
    sample_params["response_format"] = None
    return TextChatParameters(**sample_params)


# Construct the application's one model only if all environment configuration is present.
watsonx_configuration: WatsonxConfiguration = initialize_watsonx_from_environment(
    credentials_factory=Credentials,
    parameters_factory=_create_watsonx_parameters,
    model_factory=ModelInference,
    configure_client=configure_structured_client,
)
llm_base = watsonx_configuration.model


def _configuration_error() -> str | None:
    return configuration_required_ui_message(watsonx_configuration)


def extract_text_from_pdf(pdf_file_path: str | object) -> str:
    """Preserve the PyPDF2 extraction boundary while accepting Gradio filepath values."""

    path = getattr(pdf_file_path, "name", pdf_file_path)
    if not path:
        raise ValueError("Please upload a resume PDF.")
    reader = PyPDF2.PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()


def text_to_speech_file(text_input: str) -> str:
    """Render one question with gTTS and return its independent audio filepath."""

    if not text_input or not text_input.strip():
        raise ValueError("Cannot create audio for an empty interview question.")
    descriptor, audio_file_path = tempfile.mkstemp(prefix="interview_question_", suffix=".mp3")
    os.close(descriptor)
    gTTS(text=text_input, lang="en").save(audio_file_path)
    return audio_file_path


def transcribe_audio_faster_whisper(
    audio_file_path: str | None,
    model_size: str = "base",
    device: str = "auto",
    compute_type: str = "auto",
) -> str:
    """Preserve Faster Whisper microphone transcription behavior."""

    if audio_file_path is None:
        return ""
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    device = "cpu"
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        segments, _ = model.transcribe(audio_file_path, beam_size=5)
        return "".join(segment.text for segment in segments).strip()
    except Exception as error:
        return f"❌ An error occurred during transcription: {error}"


def _format_report(report: object) -> str:
    return report.model_dump_json(indent=2)


def start_interview(
    resume_path: str | object,
    job_description: str,
    total_number: int,
) -> tuple[str, str | None, None, str, str, InterviewApplicationSession | None]:
    """Build profiles/matches and start one new engine-backed Gradio session."""

    error = _configuration_error()
    if error:
        return "", None, None, error, "Start Interview", None
    update = start_application_interview(
        resume_path,
        job_description,
        total_number,
        extract_pdf=extract_text_from_pdf,
        build_candidate=build_candidate_profile,
        build_role=build_role_profile,
        create_session=create_interview_session,
        text_to_speech=text_to_speech_file,
    )
    return update.question_text, update.question_audio_path, None, update.status, update.submit_label, update.session


def submit_answer(
    session: InterviewApplicationSession | None,
    answer_audio_path: str | None,
) -> tuple[str, str | None, None, str, str, InterviewApplicationSession | None]:
    """Transcribe one answer and advance only through the session-local engine."""

    error = _configuration_error()
    if error:
        return "", None, None, error, "Start Interview", None
    update = submit_application_answer(
        session,
        answer_audio_path,
        transcribe=transcribe_audio_faster_whisper,
        text_to_speech=text_to_speech_file,
        format_report=_format_report,
    )
    return update.question_text, update.question_audio_path, None, update.status, update.submit_label, update.session


def end_interview(
    session: InterviewApplicationSession | None,
) -> tuple[str, str | None, None, str, str, InterviewApplicationSession | None]:
    error = _configuration_error()
    if error:
        return "", None, None, error, "Start Interview", None
    update = end_application_interview(session, format_report=_format_report)
    return update.question_text, update.question_audio_path, None, update.status, update.submit_label, update.session


with gr.Blocks() as demo:
    gr.Markdown("# Personalized Interview Coach")
    gr.Markdown("## Upload your PDF resume/CV and paste the job description.")
    if not watsonx_configuration.is_ready:
        gr.Markdown(f"⚠️ **configuration_required** — {CONFIGURATION_REQUIRED_MESSAGE}")
    session_state = gr.State(value=None)
    with gr.Row():
        resume_input = gr.File(label="Upload Resume (PDF)", type="filepath")
        job_desc_input = gr.Textbox(label="Job Description", lines=15)
    num_q_input = gr.Slider(label="Number of Questions", minimum=1, maximum=10, value=5, step=1)
    start_btn = gr.Button("Start Interview", scale=2, min_width=200)
    interviewer_question_text = gr.Textbox(label="Interviewer Question", lines=4, interactive=False)
    interviewer_question_audio = gr.Audio(label="Interviewer Question Audio", type="filepath")
    user_answer = gr.Audio(sources=["microphone"], type="filepath", label="Your turn! Record Your Answer.")
    submit_btn = gr.Button("Submit Answer", scale=2, min_width=200)
    end_btn = gr.Button("End Interview Early")
    evaluation_textbox = gr.Textbox(label="Interview Status / Final Report", lines=20)

    start_btn.click(
        fn=start_interview,
        inputs=[resume_input, job_desc_input, num_q_input],
        outputs=[interviewer_question_text, interviewer_question_audio, user_answer, evaluation_textbox, submit_btn, session_state],
    )
    submit_btn.click(
        fn=submit_answer,
        inputs=[session_state, user_answer],
        outputs=[interviewer_question_text, interviewer_question_audio, user_answer, evaluation_textbox, submit_btn, session_state],
    )
    end_btn.click(
        fn=end_interview,
        inputs=[session_state],
        outputs=[interviewer_question_text, interviewer_question_audio, user_answer, evaluation_textbox, submit_btn, session_state],
    )


if __name__ == "__main__":
    demo.launch(share=True)
