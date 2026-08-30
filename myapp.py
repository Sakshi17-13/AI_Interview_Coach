import os
import html
import tempfile
from pathlib import Path

import gradio as gr
import PyPDF2

from dotenv import load_dotenv
from gtts import gTTS
from faster_whisper import WhisperModel

from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters
from ibm_watsonx_ai import Credentials

# Existing application layer
from application_controller import (
    start_application_interview,
    submit_application_answer,
    end_application_interview,
)


# ============================================================
# 1. LOAD CONFIGURATION
# ============================================================

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID", "skills-network")

IBM_URL = os.getenv(
    "IBM_URL",
    "https://us-south.ml.cloud.ibm.com"
)

MODEL_ID = os.getenv(
    "MODEL_ID",
    "meta-llama/llama-3-3-70b-instruct"
)

WHISPER_MODEL_SIZE = os.getenv(
    "WHISPER_MODEL",
    "base"
)

try:
    MAX_QUESTIONS = int(
        os.getenv("MAX_QUESTIONS", "10")
    )
except ValueError:
    MAX_QUESTIONS = 10

IBM_API_KEY = os.getenv("IBM_CLOUD_API_KEY")

if not IBM_API_KEY:
    raise RuntimeError(
        "IBM_CLOUD_API_KEY is missing from your .env file. "
        "Add your IBM Cloud API key and restart the application."
    )


# ============================================================
# 2. EXISTING WATSONX CONFIGURATION
# ============================================================

credentials = Credentials(
    url=IBM_URL,
    api_key=IBM_API_KEY
)

sample_params = TextChatParameters.get_sample_params()
sample_params["max_tokens"] = 2000
sample_params["response_format"] = None

params = TextChatParameters(
    **sample_params
)

llm_base = ModelInference(
    model_id=MODEL_ID,
    credentials=credentials,
    project_id=PROJECT_ID,
    params=params,
)


# ============================================================
# 3. WHISPER
# ============================================================

_whisper_model = None


def get_whisper_model():
    """Load Faster Whisper once per application process."""

    global _whisper_model

    if _whisper_model is None:
        _whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device="cpu",
            compute_type="int8"
        )

    return _whisper_model


# ============================================================
# 4. SESSION STATE
# ============================================================

def create_initial_state():
    """
    UI session state.

    The actual interview state belongs to the application
    controller/session object.
    """

    return {
        "session": None,
        "started": False,
        "finished": False,
        "step": 0,
        "total_questions": 5,
        "current_question": "",
        "current_stage": "",
        "history": [],
        "last_audio": None,
        "report": None,
    }


# ============================================================
# 5. PDF EXTRACTION
# ============================================================

def extract_text_from_pdf(pdf_file_path):
    """
    Extract resume text at the application/UI boundary.
    """

    if not pdf_file_path:
        raise ValueError(
            "Please upload a PDF resume."
        )

    try:
        reader = PyPDF2.PdfReader(
            pdf_file_path
        )

        text_parts = []

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text_parts.append(
                    page_text
                )

        text = "\n".join(
            text_parts
        ).strip()

        if not text:
            raise ValueError(
                "We couldn't find readable text in this resume."
            )

        return text

    except ValueError:
        raise

    except Exception:
        raise ValueError(
            "Unable to read this resume. "
            "Please upload a valid PDF."
        )


# ============================================================
# 6. SPEECH TO TEXT
# ============================================================

def transcribe_audio(audio_file_path):

    if not audio_file_path:
        return ""

    try:
        model = get_whisper_model()

        segments, _ = model.transcribe(
            audio_file_path,
            beam_size=5
        )

        # IMPORTANT:
        # Preserve the transcript text returned by Whisper.
        transcript = " ".join(
            segment.text
            for segment in segments
        )

        return transcript

    except Exception:
        raise RuntimeError(
            "We couldn't transcribe that recording. "
            "Please record your answer again."
        )


# ============================================================
# 7. TEXT TO SPEECH
# ============================================================

def text_to_speech_file(text):

    if not text:
        return None

    try:

        temp_dir = Path(
            tempfile.gettempdir()
        )

        # Unique file name so sessions/questions
        # don't overwrite each other.
        import uuid

        audio_path = (
            temp_dir /
            f"interview_question_{uuid.uuid4().hex}.mp3"
        )

        tts = gTTS(
            text=text,
            lang="en"
        )

        tts.save(
            str(audio_path)
        )

        return str(audio_path)

    except Exception:
        raise RuntimeError(
            "We couldn't generate the question audio."
        )


# ============================================================
# 8. SAFE HTML
# ============================================================

def safe_html(value):
    if value is None:
        return ""

    return html.escape(
        str(value)
    )


# ============================================================
# 9. PROGRESS BAR
# ============================================================

def create_progress_html(
    current,
    total
):

    if total <= 0:
        total = 1

    current = max(
        0,
        min(current, total)
    )

    percentage = int(
        (current / total) * 100
    )

    return f"""
    <div class="progress-wrapper">

        <div class="progress-header">
            <span>Interview Progress</span>
            <strong>{current} / {total}</strong>
        </div>

        <div class="progress-track">
            <div
                class="progress-fill"
                style="width: {percentage}%;">
            </div>
        </div>

        <div class="progress-percent">
            {percentage}% complete
        </div>

    </div>
    """


# ============================================================
# 10. STAGE DISPLAY
# ============================================================

def format_stage(stage):

    if not stage:
        return "Interview"

    stage = str(stage).replace(
        "_",
        " "
    )

    return stage.title()


# ============================================================
# 11. RESULT DASHBOARD
# ============================================================

def get_report_value(
    report,
    name,
    default=None
):

    if report is None:
        return default

    if isinstance(report, dict):
        return report.get(
            name,
            default
        )

    return getattr(
        report,
        name,
        default
    )


def create_result_dashboard(report):

    if report is None:
        return """
        <div class="empty-result">
            Your final interview results will appear here.
        </div>
        """

    overall = get_report_value(
        report,
        "overall_score",
        None
    )

    job_fit = get_report_value(
        report,
        "job_fit_score",
        None
    )

    recommendation = get_report_value(
        report,
        "recommendation",
        None
    )

    dimensions = get_report_value(
        report,
        "dimension_scores",
        None
    )

    strengths = get_report_value(
        report,
        "strengths",
        None
    )

    weaknesses = get_report_value(
        report,
        "weaknesses",
        None
    )

    improvement = get_report_value(
        report,
        "improvement_recommendations",
        None
    )

    if overall is None:
        overall_display = "—"
    else:
        overall_display = f"{overall:.0f}"

    if job_fit is None:
        job_fit_display = "—"
    else:
        job_fit_display = f"{job_fit:.0f}"

    recommendation_text = (
        safe_html(recommendation)
        if recommendation
        else "Not available"
    )

    def render_dimension(
        label,
        value
    ):

        if value is None:
            return ""

        try:
            numeric = float(value)
        except Exception:
            return ""

        numeric = max(
            0,
            min(numeric, 100)
        )

        return f"""
        <div class="metric-row">

            <div class="metric-label">
                <span>{label}</span>
                <strong>{numeric:.0f}%</strong>
            </div>

            <div class="metric-track">
                <div
                    class="metric-fill"
                    style="width:{numeric}%;">
                </div>
            </div>

        </div>
        """

    dimension_html = ""

    if dimensions:

        if isinstance(
            dimensions,
            dict
        ):

            mapping = {
                "technical_accuracy":
                    "Technical Accuracy",
                "relevance":
                    "Relevance",
                "specificity":
                    "Specificity",
                "communication":
                    "Communication",
            }

            for key, label in mapping.items():

                value = dimensions.get(
                    key
                )

                dimension_html += render_dimension(
                    label,
                    value
                )

    def render_list(
        title,
        items
    ):

        if not items:
            return ""

        if isinstance(
            items,
            str
        ):
            items = [
                items
            ]

        rows = ""

        for item in items:
            rows += f"""
            <li>{safe_html(item)}</li>
            """

        return f"""
        <div class="result-section">

            <h3>{title}</h3>

            <ul>
                {rows}
            </ul>

        </div>
        """

    strengths_html = render_list(
        "Strengths",
        strengths
    )

    weaknesses_html = render_list(
        "Areas to Improve",
        weaknesses
    )

    improvement_html = render_list(
        "Recommended Practice",
        improvement
    )

    return f"""
    <div class="results-container">

        <div class="result-hero">

            <div>
                <div class="result-label">
                    OVERALL PERFORMANCE
                </div>

                <div class="result-score">
                    {overall_display}
                    <span>/100</span>
                </div>
            </div>

            <div class="fit-card">

                <div class="result-label">
                    JOB FIT
                </div>

                <div class="fit-score">
                    {job_fit_display}
                    <span>/100</span>
                </div>

            </div>

        </div>


        <div class="recommendation-card">

            <div class="result-label">
                RECOMMENDATION
            </div>

            <div class="recommendation">
                {recommendation_text}
            </div>

        </div>


        <div class="result-section">

            <h3>Performance Breakdown</h3>

            {dimension_html}

        </div>


        {strengths_html}

        {weaknesses_html}

        {improvement_html}

    </div>
    """


# ============================================================
# 12. SETUP HANDLER
# ============================================================

def ui_start_interview(
    resume_file,
    job_description,
    total_questions,
    state
):

    state = (
        state
        if state
        else create_initial_state()
    )

    try:

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if not resume_file:

            return (
                state,
                gr.update(
                    visible=True
                ),
                gr.update(
                    value="❌ Please upload your resume PDF."
                ),
                gr.update(
                    value=create_progress_html(
                        0,
                        int(total_questions or 5)
                    )
                ),
                "",
                "",
                None,
                gr.update(
                    interactive=True
                ),
                gr.update(
                    interactive=True
                ),
                gr.update(
                    interactive=False
                ),
                gr.update(
                    interactive=False
                ),
                gr.update(
                    visible=False
                ),
                gr.update(
                    visible=False
                ),
            )

        if not job_description or not job_description.strip():

            return (
                state,
                gr.update(
                    visible=True
                ),
                gr.update(
                    value="❌ Please provide a job description before starting."
                ),
                gr.update(
                    value=create_progress_html(
                        0,
                        int(total_questions or 5)
                    )
                ),
                "",
                "",
                None,
                gr.update(
                    interactive=True
                ),
                gr.update(
                    interactive=True
                ),
                gr.update(
                    interactive=False
                ),
                gr.update(
                    interactive=False
                ),
                gr.update(
                    visible=False
                ),
                gr.update(
                    visible=False
                ),
            )

        total_questions = int(
            total_questions
        )

        total_questions = max(
            1,
            min(
                total_questions,
                MAX_QUESTIONS
            )
        )

        # ----------------------------------------------------
        # Resume extraction
        # ----------------------------------------------------

        status = "📄 Reading your resume..."

        # The status is returned after the backend call because
        # Gradio event functions are synchronous.
        resume_text = extract_text_from_pdf(
            resume_file
        )

        # ----------------------------------------------------
        # Existing application controller
        # ----------------------------------------------------

        status = (
            "🔍 Analyzing your profile "
            "and preparing the interview..."
        )

        application_update = (
            start_application_interview(
                resume_file,
                job_description,
                total_questions
            )
        )

        session = getattr(
            application_update,
            "session",
            None
        )

        if session is None:

            return (
                state,
                gr.update(
                    visible=True
                ),
                gr.update(
                    value="❌ We couldn't prepare the interview. Please try again."
                ),
                gr.update(
                    value=create_progress_html(
                        0,
                        total_questions
                    )
                ),
                "",
                "",
                None,
                gr.update(
                    interactive=True
                ),
                gr.update(
                    interactive=True
                ),
                gr.update(
                    interactive=False
                ),
                gr.update(
                    interactive=False
                ),
                gr.update(
                    visible=False
                ),
                gr.update(
                    visible=False
                ),
            )

        question_text = getattr(
            application_update,
            "question_text",
            ""
        )

        question_audio = getattr(
            application_update,
            "question_audio_path",
            None
        )

        # ----------------------------------------------------
        # Fallback TTS if controller did not provide audio
        # ----------------------------------------------------

        if question_text and not question_audio:

            question_audio = (
                text_to_speech_file(
                    question_text
                )
            )

        stage = ""

        question_object = getattr(
            session.engine,
            "get_current_question",
            lambda: None
        )()

        if question_object:

            stage = getattr(
                question_object,
                "stage",
                ""
            )

        state = create_initial_state()

        state["session"] = session
        state["started"] = True
        state["finished"] = False
        state["step"] = 1
        state["total_questions"] = total_questions
        state["current_question"] = question_text
        state["current_stage"] = stage
        state["history"] = []

        return (
            state,

            gr.update(
                visible=True
            ),

            gr.update(
                value="Interview started. Take your time and answer naturally."
            ),

            gr.update(
                value=create_progress_html(
                    1,
                    total_questions
                )
            ),

            question_text,

            format_stage(stage),

            question_audio,

            gr.update(
                interactive=False
            ),

            gr.update(
                interactive=False
            ),

            gr.update(
                interactive=True
            ),

            gr.update(
                interactive=True
            ),

            gr.update(
                visible=True
            ),

            gr.update(
                visible=True
            ),
        )

    except ValueError as e:

        return (
            state,
            gr.update(
                visible=True
            ),
            gr.update(
                value=f"❌ {str(e)}"
            ),
            gr.update(
                value=create_progress_html(
                    0,
                    int(total_questions or 5)
                )
            ),
            "",
            "",
            None,
            gr.update(
                interactive=True
            ),
            gr.update(
                interactive=True
            ),
            gr.update(
                interactive=False
            ),
            gr.update(
                interactive=False
            ),
            gr.update(
                visible=False
            ),
            gr.update(
                visible=False
            ),
        )

    except Exception:

        return (
            state,
            gr.update(
                visible=True
            ),
            gr.update(
                value="❌ We couldn't prepare the interview. Please check your inputs and try again."
            ),
            gr.update(
                value=create_progress_html(
                    0,
                    int(total_questions or 5)
                )
            ),
            "",
            "",
            None,
            gr.update(
                interactive=True
            ),
            gr.update(
                interactive=True
            ),
            gr.update(
                interactive=False
            ),
            gr.update(
                interactive=False
            ),
            gr.update(
                visible=False
            ),
            gr.update(
                visible=False
            ),
        )


# ============================================================
# 13. SUBMIT ANSWER HANDLER
# ============================================================

def ui_submit_answer(
    audio_answer,
    state
):

    if not state or not state.get(
        "started"
    ):

        return (
            state,
            None,
            "",
            "",
            gr.update(value="❌ Please start the interview first."),
            gr.update(value=create_progress_html(0, 1)),
            "",
            "",
            gr.update(interactive=False),
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(visible=False),
        )

    session = state.get(
        "session"
    )

    if session is None:

        return (
            state,
            None,
            "",
            "",
            gr.update(value="❌ Your interview session could not be found. Please start a new interview."),
            gr.update(value=create_progress_html(0, 1)),
            "",
            "",
            gr.update(interactive=False),
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(visible=False),
        )

    if not audio_answer:

        return (
            state,
            None,
            state.get(
                "current_question",
                ""
            ),
            format_stage(
                state.get(
                    "current_stage",
                    ""
                )
            ),
            gr.update(
                value="⚠️ Please record your answer before submitting."
            ),
            gr.update(
                value=create_progress_html(
                    state.get(
                        "step",
                        1
                    ),
                    state.get(
                        "total_questions",
                        1
                    )
                )
            ),
            "",
            "",
            gr.update(
                interactive=True
            ),
            gr.update(
                interactive=True
            ),
            gr.update(
                interactive=True
            ),
            gr.update(
                visible=False
            ),
        )

    try:

        # ----------------------------------------------------
        # STT
        # ----------------------------------------------------

        transcript = transcribe_audio(
            audio_answer
        )

        if not transcript:

            raise RuntimeError(
                "We couldn't transcribe that recording. "
                "Please record your answer again."
            )

        # ----------------------------------------------------
        # Application controller
        # ----------------------------------------------------

        application_update = (
            submit_application_answer(
                session,
                audio_answer
            )
        )

        # ----------------------------------------------------
        # Session may have been updated
        # ----------------------------------------------------

        new_session = getattr(
            application_update,
            "session",
            None
        )

        if new_session is not None:
            session = new_session
            state["session"] = session

        question_text = getattr(
            application_update,
            "question_text",
            ""
        )

        question_audio = getattr(
            application_update,
            "question_audio_path",
            None
        )

        label = getattr(
            application_update,
            "status_text",
            ""
        )

        button_label = getattr(
            application_update,
            "button_label",
            ""
        )

        # ----------------------------------------------------
        # Completion
        # ----------------------------------------------------

        report = None

        if hasattr(
            session,
            "build_final_report"
        ):

            try:

                if session.engine.is_complete():

                    report = (
                        session.build_final_report()
                    )

            except Exception:
                report = None

        if report is not None:

            state["finished"] = True
            state["report"] = report

            result_html = (
                create_result_dashboard(
                    report
                )
            )

            return (
                state,
                None,
                "Interview Completed",
                "Closing",
                gr.update(
                    value=(
                        "🎉 Interview completed. "
                        "Your personalized results are ready."
                    )
                ),
                gr.update(
                    value=create_progress_html(
                        state.get(
                            "total_questions",
                            1
                        ),
                        state.get(
                            "total_questions",
                            1
                        )
                    )
                ),
                transcript,
                result_html,
                gr.update(
                    interactive=False
                ),
                gr.update(
                    interactive=False
                ),
                gr.update(
                    interactive=False
                ),
                gr.update(
                    visible=True
                ),
            )

        # ----------------------------------------------------
        # Next question
        # ----------------------------------------------------

        if question_text:

            if not question_audio:

                question_audio = (
                    text_to_speech_file(
                        question_text
                    )
                )

            state["step"] += 1
            state["current_question"] = (
                question_text
            )

            question_object = None

            try:

                question_object = (
                    session.engine.get_current_question()
                )

            except Exception:
                pass

            stage = ""

            if question_object:

                stage = getattr(
                    question_object,
                    "stage",
                    ""
                )

            state["current_stage"] = stage

            progress_html = (
                create_progress_html(
                    state["step"],
                    state["total_questions"]
                )
            )

            status_message = (
                label
                if label
                else "Answer recorded. Continue with the next question."
            )

            return (
                state,
                question_audio,
                question_text,
                format_stage(stage),
                gr.update(
                    value=status_message
                ),
                gr.update(
                    value=progress_html
                ),
                transcript,
                "",
                gr.update(
                    interactive=True
                ),
                gr.update(
                    interactive=True
                ),
                gr.update(
                    interactive=True
                ),
                gr.update(
                    visible=False
                ),
            )

        # ----------------------------------------------------
        # No next question / retry situation
        # ----------------------------------------------------

        return (
            state,
            None,
            state.get(
                "current_question",
                ""
            ),
            format_stage(
                state.get(
                    "current_stage",
                    ""
                )
            ),
            gr.update(
                value=(
                    label
                    if label
                    else "Your answer was recorded. Please try the next step."
                )
            ),
            gr.update(
                value=create_progress_html(
                    state.get(
                        "step",
                        1
                    ),
                    state.get(
                        "total_questions",
                        1
                    )
                )
            ),
            transcript,
            "",
            gr.update(
                interactive=True
            ),
            gr.update(
                interactive=True
            ),
            gr.update(
                interactive=True
            ),
            gr.update(
                visible=False
            ),
        )

    except Exception:

        return (
            state,
            None,
            state.get(
                "current_question",
                ""
            ),
            format_stage(
                state.get(
                    "current_stage",
                    ""
                )
            ),
            gr.update(
                value=(
                    "❌ We couldn't analyze that answer. "
                    "Your interview progress is محفوظ. "
                    "Please try again."
                )
            ),
            gr.update(
                value=create_progress_html(
                    state.get(
                        "step",
                        1
                    ),
                    state.get(
                        "total_questions",
                        1
                    )
                )
            ),
            transcript if "transcript" in locals() else "",
            "",
            gr.update(
                interactive=True
            ),
            gr.update(
                interactive=True
            ),
            gr.update(
                interactive=True
            ),
            gr.update(
                visible=False
            ),
        )


# ============================================================
# 14. END INTERVIEW
# ============================================================

def ui_end_interview(
    state
):

    if not state or not state.get(
        "started"
    ):

        return (
            state,
            gr.update(
                value="❌ No active interview."
            ),
            gr.update(
                visible=False
            ),
            gr.update(
                interactive=False
            ),
            gr.update(
                interactive=False
            ),
            gr.update(
                interactive=True
            ),
        )

    session = state.get(
        "session"
    )

    if session is None:

        return (
            state,
            gr.update(
                value="❌ Interview session not found."
            ),
            gr.update(
                visible=False
            ),
            gr.update(
                interactive=False
            ),
            gr.update(
                interactive=False
            ),
            gr.update(
                interactive=True
            ),
        )

    try:

        application_update = (
            end_application_interview(
                session
            )
        )

        new_session = getattr(
            application_update,
            "session",
            None
        )

        if new_session is not None:

            session = new_session
            state["session"] = session

        report = None

        if hasattr(
            session,
            "end_early"
        ):

            report = (
                session.end_early()
            )

        state["finished"] = True
        state["report"] = report

        result_html = (
            create_result_dashboard(
                report
            )
        )

        return (
            state,

            gr.update(
                value=(
                    "Interview ended. "
                    "Your current progress has been saved."
                )
            ),

            gr.update(
                value=result_html,
                visible=True
            ),

            gr.update(
                interactive=False
            ),

            gr.update(
                interactive=False
            ),

            gr.update(
                interactive=True
            ),
        )

    except Exception:

        return (
            state,
            gr.update(
                value=(
                    "❌ We couldn't prepare your final report. "
                    "Please try again."
                )
            ),
            gr.update(
                visible=False
            ),
            gr.update(
                interactive=True
            ),
            gr.update(
                interactive=True
            ),
            gr.update(
                interactive=True
            ),
        )


# ============================================================
# 15. RESET
# ============================================================

def reset_interview():

    state = create_initial_state()

    return (
        state,

        None,

        "",

        "",

        "",

        "",

        "Ready to start",

        gr.update(
            value=create_progress_html(
                0,
                5
            )
        ),

        "No transcript yet.",

        gr.update(
            visible=False
        ),

        gr.update(
            visible=False
        ),

        gr.update(
            interactive=True
        ),

        gr.update(
            interactive=True
        ),

        gr.update(
            interactive=False
        ),

        gr.update(
            interactive=False
        ),

        gr.update(
            visible=False
        ),
    )


# ============================================================
# 16. CUSTOM CSS
# ============================================================

CUSTOM_CSS = """

/* ---------------------------------------------------------
   GLOBAL
--------------------------------------------------------- */

.gradio-container {
    max-width: 1180px !important;
    margin: auto !important;
}

body {
    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
}

.title {
    text-align: center;
    margin-top: 18px;
    margin-bottom: 4px;
}

.subtitle {
    text-align: center;
    opacity: 0.72;
    font-size: 16px;
    margin-bottom: 18px;
}


/* ---------------------------------------------------------
   CARDS
--------------------------------------------------------- */

.setup-card,
.interview-card,
.status-card,
.results-card {
    border-radius: 18px !important;
}


/* ---------------------------------------------------------
   PROGRESS
--------------------------------------------------------- */

.progress-wrapper {
    width: 100%;
    padding: 8px 0 15px 0;
}

.progress-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 16px;
    margin-bottom: 8px;
}

.progress-track {
    height: 12px;
    border-radius: 999px;
    background: rgba(128,128,128,0.20);
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(
        90deg,
        #4f46e5,
        #06b6d4
    );
    transition: width 0.3s ease;
}

.progress-percent {
    margin-top: 7px;
    text-align: right;
    font-size: 12px;
    opacity: 0.65;
}


/* ---------------------------------------------------------
   QUESTION
--------------------------------------------------------- */

.question-box {
    padding: 22px;
    border-radius: 16px;
    border: 1px solid rgba(128,128,128,0.22);
    min-height: 140px;
}

.question-label {
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    opacity: 0.65;
    margin-bottom: 10px;
}

.question-stage {
    display: inline-block;
    padding: 5px 12px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 14px;
    background: rgba(79,70,229,0.12);
}

.question-text {
    font-size: 23px;
    line-height: 1.5;
    font-weight: 600;
}


/* ---------------------------------------------------------
   RESULTS
--------------------------------------------------------- */

.results-container {
    padding: 8px;
}

.result-hero {
    display: flex;
    gap: 18px;
    align-items: stretch;
    margin-bottom: 18px;
}

.result-hero > div:first-child {
    flex: 1;
}

.fit-card {
    min-width: 210px;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid rgba(128,128,128,0.22);
}

.result-label {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.08em;
    opacity: 0.65;
}

.result-score {
    font-size: 68px;
    font-weight: 800;
    line-height: 1.1;
    margin-top: 7px;
}

.result-score span,
.fit-score span {
    font-size: 22px;
    opacity: 0.6;
}

.fit-score {
    font-size: 40px;
    font-weight: 800;
    margin-top: 10px;
}

.recommendation-card {
    padding: 20px;
    border-radius: 16px;
    margin-bottom: 18px;
    border: 1px solid rgba(128,128,128,0.22);
}

.recommendation {
    font-size: 24px;
    font-weight: 700;
    margin-top: 7px;
}

.result-section {
    padding: 20px;
    border-radius: 16px;
    margin-bottom: 18px;
    border: 1px solid rgba(128,128,128,0.18);
}

.result-section h3 {
    margin-top: 0;
}

.metric-row {
    margin-bottom: 16px;
}

.metric-label {
    display: flex;
    justify-content: space-between;
    margin-bottom: 6px;
}

.metric-track {
    height: 9px;
    border-radius: 999px;
    background: rgba(128,128,128,0.18);
    overflow: hidden;
}

.metric-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(
        90deg,
        #4f46e5,
        #06b6d4
    );
}

.result-section ul {
    padding-left: 22px;
    line-height: 1.8;
}

.empty-result {
    padding: 30px;
    text-align: center;
    opacity: 0.65;
}


/* ---------------------------------------------------------
   RESPONSIVE
--------------------------------------------------------- */

@media (max-width: 700px) {

    .result-hero {
        flex-direction: column;
    }

    .fit-card {
        min-width: auto;
    }

    .question-text {
        font-size: 19px;
    }

    .result-score {
        font-size: 52px;
    }

}

"""


# ============================================================
# 17. GRADIO APPLICATION
# ============================================================

with gr.Blocks(
    title="AI Interview Coach",
    css=CUSTOM_CSS,
    theme=gr.themes.Soft()
) as demo:

    # --------------------------------------------------------
    # SESSION STATE
    # --------------------------------------------------------

    session_state = gr.State(
        create_initial_state()
    )


    # ========================================================
    # HEADER
    # ========================================================

    gr.Markdown(
        "# 🎤 AI Interview Coach",
        elem_classes="title"
    )

    gr.Markdown(
        "Practice a personalized mock interview based on your "
        "resume and target job.",
        elem_classes="subtitle"
    )


    # ========================================================
    # SETUP SCREEN
    # ========================================================

    with gr.Group(
        elem_classes="setup-card"
    ):

        gr.Markdown(
            "## Interview Setup"
        )

        gr.Markdown(
            "Upload your resume, provide the target job description, "
            "and choose how many questions you want to practice."
        )

        with gr.Row():

            with gr.Column(
                scale=1
            ):

                resume_input = gr.File(
                    label="Resume",
                    file_types=[".pdf"],
                    type="filepath"
                )

            with gr.Column(
                scale=1
            ):

                job_desc_input = gr.Textbox(
                    label="Job Description",
                    placeholder=(
                        "Paste the job description here..."
                    ),
                    lines=10
                )

        with gr.Row():

            num_q_input = gr.Slider(
                label="Number of Questions",
                minimum=1,
                maximum=MAX_QUESTIONS,
                value=5,
                step=1
            )

        with gr.Row():

            start_btn = gr.Button(
                "Start Interview",
                variant="primary",
                size="lg"
            )

            reset_btn = gr.Button(
                "Start New Interview",
                size="lg"
            )


    # ========================================================
    # INTERVIEW AREA
    # ========================================================

    interview_area = gr.Group(
        visible=False
    )

    with interview_area:

        gr.Markdown(
            "## Interview Dashboard"
        )

        progress_html = gr.HTML(
            create_progress_html(
                0,
                5
            )
        )

        with gr.Group(
            elem_classes="interview-card"
        ):

            with gr.Row():

                with gr.Column(
                    scale=2
                ):

                    question_stage = gr.Markdown(
                        "Interview",
                    )

                    question_text = gr.Textbox(
                        label="Question",
                        lines=5,
                        interactive=False,
                        placeholder=(
                            "Your interview question will appear here."
                        )
                    )

                    gr.Markdown(
                        "### 🔊 Question Audio"
                    )

                    interviewer_audio = gr.Audio(
                        label="Listen to the question",
                        type="filepath",
                        autoplay=False
                    )

                with gr.Column(
                    scale=1
                ):

                    gr.Markdown(
                        "### 🎙 Your Answer"
                    )

                    user_answer = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        label="Record your answer"
                    )

                    submit_btn = gr.Button(
                        "Submit Answer",
                        variant="primary",
                        size="lg",
                        interactive=False
                    )

                    end_btn = gr.Button(
                        "End Interview",
                        variant="stop",
                        interactive=False
                    )


        status_text = gr.Markdown(
            "Ready."
        )


        # ----------------------------------------------------
        # TRANSCRIPT
        # ----------------------------------------------------

        with gr.Accordion(
            "Conversation Transcript",
            open=False
        ):

            transcript_display = gr.Markdown(
                "No transcript yet."
            )


    # ========================================================
    # RESULTS
    # ========================================================

    results_area = gr.Group(
        visible=False
    )

    with results_area:

        gr.Markdown(
            "## Interview Results"
        )

        results_dashboard = gr.HTML(
            create_result_dashboard(
                None
            )
        )

        new_interview_btn = gr.Button(
            "Start New Interview",
            variant="primary",
            size="lg"
        )


    # ========================================================
    # START EVENT
    # ========================================================

    start_btn.click(
        fn=ui_start_interview,
        inputs=[
            resume_input,
            job_desc_input,
            num_q_input,
            session_state
        ],
        outputs=[
            session_state,
            interview_area,
            status_text,
            progress_html,
            question_text,
            question_stage,
            interviewer_audio,
            start_btn,
            reset_btn,
            submit_btn,
            end_btn,
            results_area,
            transcript_display,
        ]
    )


    # ========================================================
    # SUBMIT ANSWER EVENT
    # ========================================================

    submit_btn.click(
        fn=ui_submit_answer,
        inputs=[
            user_answer,
            session_state
        ],
        outputs=[
            session_state,
            interviewer_audio,
            question_text,
            question_stage,
            status_text,
            progress_html,
            transcript_display,
            results_dashboard,
            submit_btn,
            end_btn,
            start_btn,
            results_area,
        ]
    )


    # ========================================================
    # END INTERVIEW EVENT
    # ========================================================

    end_btn.click(
        fn=ui_end_interview,
        inputs=[
            session_state
        ],
        outputs=[
            session_state,
            status_text,
            results_dashboard,
            submit_btn,
            end_btn,
            new_interview_btn,
        ]
    )


    # ========================================================
    # RESET EVENTS
    # ========================================================

    reset_outputs = [
        session_state,
        interviewer_audio,
        question_text,
        question_stage,
        status_text,
        progress_html,
        transcript_display,
        results_area,
        results_dashboard,
        start_btn,
        reset_btn,
        submit_btn,
        end_btn,
        interview_area,
        user_answer,
    ]

    reset_btn.click(
        fn=reset_interview,
        inputs=[],
        outputs=[
            session_state,
            interviewer_audio,
            question_text,
            question_stage,
            status_text,
            progress_html,
            transcript_display,
            results_area,
            results_dashboard,
            start_btn,
            reset_btn,
            submit_btn,
            end_btn,
            interview_area,
            user_answer,
        ]
    )

    new_interview_btn.click(
        fn=reset_interview,
        inputs=[],
        outputs=[
            session_state,
            interviewer_audio,
            question_text,
            question_stage,
            status_text,
            progress_html,
            transcript_display,
            results_area,
            results_dashboard,
            start_btn,
            reset_btn,
            submit_btn,
            end_btn,
            interview_area,
            user_answer,
        ]
    )


# ============================================================
# 18. APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("AI INTERVIEW COACH")
    print("=" * 60)
    print(f"Model: {MODEL_ID}")
    print(f"Whisper: {WHISPER_MODEL_SIZE}")
    print(f"Maximum Questions: {MAX_QUESTIONS}")
    print("=" * 60)

    demo.launch(
        share=True,
        show_error=False
    )