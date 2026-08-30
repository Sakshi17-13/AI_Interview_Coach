# AI and Application Testing

## Current result

The offline unittest suite currently contains **85 passing tests**. It uses fake structured-model responses and fake application dependencies; it does not require Watsonx credentials, a browser, microphone hardware, gTTS network access, or Faster Whisper model downloads.

Run from the repository root:

```powershell
python -m unittest discover -s tests -v
python -m compileall ai tests
git diff --check
```

## Covered behavior

| Area | Test files | Verified behavior |
| --- | --- | --- |
| Profile building | `test_profile_builder.py` | Valid structured profile creation from the example inputs; safe absent fields; resume evidence grounding; unsupported profile claims rejected; required/preferred role IDs cannot overlap. |
| Semantic skill matching | `test_skill_matcher.py` | Strong semantic Python match, partial match, missing Docker as not demonstrated, fabricated candidate/job evidence rejection, required/preferred separation, weighted scoring, empty candidate/role handling, invalid positive-match evidence, duplicate IDs. |
| Adaptive planning | `test_interview_planner.py` | Introduction/closing selection, required and weak priority, strong competency coverage, lower priority for covered topics, low-score follow-ups, follow-up limit, target count, empty competencies, stable deterministic ranking, invalid state/reference rejection, separation of resume match from interview coverage. |
| Question generation | `test_question_generator.py` | Technical, behavioral, project, scenario, and follow-up wording; plan metadata fixed; fabricated details/empty output/internal metadata/multiple questions rejected; deterministic repeated output. |
| Answer assessment | `test_answer_assessor.py` | Excellent/weak/partial/behavioral/project answers; empty answer deterministic path; verbatim answer evidence; resume-only evidence rejected; score/confidence/follow-up validation; question identity copied; internal instructions rejected. |
| Interview engine | `test_interview_engine.py` | Initialization, first question, exact transcript storage, coverage update, weak follow-up, strong progression, limits/completion, empty transcript, missing question/uninitialized errors, planner/generator failure state preservation, session isolation, planner-owned metadata, deterministic priority. |
| Final report | `test_report_builder.py` | Multi-competency aggregation, deterministic dimensions/overall score, required/preferred weighting, required-weakness cap, only-required/preferred/no competency cases, blank answers, incomplete report status, qualitative evidence rejection, resume-vs-answer evidence distinction, threshold bands. |
| Application adapter | `test_interview_app_adapter.py` | Session creates match/engine, exact transcript forwarding, report handoff, multiple independent sessions. |
| Application hardening | `test_application_controller.py` | Valid setup with separate question text/audio, corrupt/empty PDF and empty JD handling, profile/match/init setup errors, blank recording, STT error, assessment retry, question-only retry, final-report retry, early termination isolation, and TTS failure retaining question text. |

## Evidence and hallucination coverage

The suite directly tests profile evidence against source material, fabricated profile/match/assessment/report evidence rejection, verbatim transcript evidence checks, and the rule that resume evidence cannot be represented as answer evidence.

## What is not covered offline

- Live watsonx responses, credentials, latency, and provider-specific error modes.
- The Gradio browser event lifecycle and serialization behavior of `gr.State`.
- Real upload behavior for malformed PDFs beyond the application-controller error path.
- Real gTTS file creation/playback/network failures and Faster Whisper microphone/model execution.
- Temporary audio-file cleanup over a long-running production session.

These need a controlled browser/runtime smoke test with valid credentials and audio hardware.
