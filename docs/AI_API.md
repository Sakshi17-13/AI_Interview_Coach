# AI/Application API

Use `interview_app_adapter.py` for normal application integration. It packages matching, engine initialization, transcript submission, and reporting into one session-local object. Store one `InterviewApplicationSession` per browser session (the current Gradio integration uses `gr.State`).

Before use, configure the existing watsonx model once:

```python
from ai import configure_structured_client
configure_structured_client(existing_model_inference)
```

Do not create another `ModelInference` instance.

## Build profiles from application input

### `ai.build_candidate_profile(resume_text) -> CandidateProfile`

- Input: extracted, non-empty resume text.
- Output: resume-grounded `CandidateProfile`.
- State change: none.
- Errors: `ValueError` for empty text, `ProfileGroundingError` for unsupported evidence, or structured-client errors for invalid model output.
- Retry: safe after correcting the input or transient model error.

### `ai.build_role_profile(job_description) -> RoleProfile`

- Input: non-empty job-description text.
- Output: evidence-grounded `RoleProfile` with required/preferred separation.
- State change/retry/errors: same pattern as candidate profile building.

Use the existing `myapp.extract_text_from_pdf(uploaded_path)` at the UI boundary. It raises PyPDF2 errors for unreadable/corrupted files and returns an empty string for PDFs with no extractable text.

## Create an interview and first question

### `create_interview_session(candidate_profile, role_profile, target_question_count, ...) -> (InterviewApplicationSession, InterviewQuestion)`

- Input: validated profiles and a positive target question count. Optional injected matcher, engine factory, report builder, and session ID are for tests/integration.
- Output: a new, independent session and the first `InterviewQuestion`.
- State change: computes a `SkillMatchReport`, initializes an `InterviewEngine`, and stores its first question.
- Errors: skill matching, planner, question-generation, or engine initialization errors propagate; no usable session is returned.
- Retry: safe by creating a new session. Do not reuse a partial setup session.

The first question text is `question.question_text`. Convert that text to audio separately; never substitute the audio filepath for the text.

## Submit an exact transcript

### `InterviewApplicationSession.submit_transcript(transcript) -> InterviewTurn`

- Input: the exact transcript returned by STT. Pass `""` for an empty recording; do not trim, normalize, or add interpretation.
- Output: `InterviewTurn(assessment, next_question, report)`.
- State change: delegates to `InterviewEngine.submit_answer`; stores transcript and assessment, updates coverage, and may advance to a next question or complete.
- Errors: raises the underlying assessment/planning/question/report error. If answer assessment itself fails, the current question remains. If next-question generation failed after a committed assessment, `needs_question_retry` is set.
- Retry: do **not** resubmit the transcript when `needs_question_retry` is true; use `retry_next_question()`. If assessment failed before commitment, transcript submission is safe to retry.

`InterviewTurn.next_question` is present while active. `InterviewTurn.report` is present only on successful completion. `InterviewTurn.assessment` is the current answer assessment.

## Detect completion and inspect state

### `session.engine.is_complete() -> bool`

- Input/output: no input; returns true only when target question count has been reached and the engine completed.
- State change/errors/retry: none; safe to call repeatedly.

### `session.engine.get_current_question() -> InterviewQuestion | None`

- Output: current question or `None` after a submitted answer before recovery/completion.
- Errors: `InterviewEngineError` if engine was never initialized.
- Retry: safe read operation.

### `session.engine.get_state() -> InterviewSessionState`

- Output: deep-copy snapshot of engine state.
- State change: none.
- Retry: safe read operation.

## Retry next question

### `session.needs_question_retry -> bool`

Signals a committed assessment whose next question could not be rendered.

### `session.retry_next_question() -> InterviewQuestion | None`

- Input: none.
- Output: recovered next question, existing current question, or `None` if complete.
- State change: retries engine question generation and clears the retry flag only when successful/completed.
- Errors: engine/planner/generator errors propagate and leave the session recoverable.
- Retry: safe to call again after a transient failure. It does not repeat answer assessment.

## Final report

### `session.build_final_report() -> FinalInterviewReport`

- Input: none.
- Output: a deterministic report plus evidence-constrained qualitative content.
- State change: none.
- Errors: report structured-output/grounding errors can propagate.
- Retry: safe; it uses the same immutable engine-state snapshot and does not resubmit an answer.

Use it after `engine.is_complete()` or after `end_early()`.

## End early

### `session.end_early() -> FinalInterviewReport`

- Input: none.
- Output: an explicitly `status="incomplete"` final report.
- State change: sets `session.ended_early = True`.
- Errors: report generation errors can propagate; the early-ended flag remains set.
- Retry: call `build_final_report()` to retry report generation. Do not call `submit_transcript()` after ending early; it raises `RuntimeError`.

Important: this does not mark the underlying engine as completed. Keep interaction behind the adapter/controller so an early-ended session cannot be advanced directly.

## Optional UI-controller entry points

`application_controller.py` exposes dependency-injected wrappers used by `myapp.py`:

| Function | Input | Output/state/error/retry |
| --- | --- | --- |
| `start_application_interview(...)` | PDF input, JD, target count, and UI dependencies | `ApplicationUpdate`; setup failures return `session=None`, so prior state is not reused. Safe to retry with corrected input. |
| `submit_application_answer(...)` | session, audio path, STT/TTS/report formatter | `ApplicationUpdate`; handles STT error, next-question retry, report retry, and preserves text/audio separately. Re-run only according to its returned label. |
| `end_application_interview(...)` | session, report formatter | `ApplicationUpdate` with incomplete report or `Retry Final Report`; safe to retry report generation. |

`ApplicationUpdate` contains `question_text`, `question_audio_path`, status text, a button label, and the session. A non-null session must be returned to the next UI event; a null session means setup did not succeed.
