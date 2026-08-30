# AI and Application Architecture

## Scope and design boundary

The project separates interview decisions from the application layer. The AI package owns extraction, matching, planning, question wording, answer assessment, state transitions, and final reporting. The application layer owns file upload, PDF text extraction, microphone/audio handling, UI state, and error presentation.

`myapp.py` creates the single existing watsonx `ModelInference` instance and passes it to `configure_structured_client`. No AI module creates a second model client.

## End-to-end flow

```text
Resume PDF --PyPDF2--> resume text --ProfileBuilder--> CandidateProfile
Job description -------------------ProfileBuilder--> RoleProfile
CandidateProfile + RoleProfile --SkillMatcher-------> SkillMatchReport
Profiles + report --InterviewEngine/Planner---------> QuestionPlan
QuestionPlan --QuestionGenerator--------------------> InterviewQuestion text
Question text --gTTS--------------------------------> separate audio filepath
Microphone audio --Faster Whisper-------------------> exact transcript
Question + transcript --AnswerAssessor--------------> AnswerAssessment
Assessment --InterviewEngine------------------------> coverage/state/next question
Completed or early-ended state --FinalReportBuilder-> FinalInterviewReport
```

## Module responsibilities

| Module | Responsibility |
| --- | --- |
| `ai/llm_client.py` | Reuses the configured watsonx model; strips code fences, parses JSON, validates Pydantic output, and makes one repair attempt. |
| `ai/prompts.py` | Bounded prompts for profile extraction, matching, planner guidance, question wording, answer assessment, and report feedback. |
| `ai/schemas.py` | Strict Pydantic contracts; unknown fields are rejected. |
| `ai/profile_builder.py` | Builds evidence-grounded candidate and role profiles, and checks extracted evidence against source text. |
| `ai/skill_matcher.py` | Semantically matches each role competency to candidate evidence and calculates match scores deterministically. |
| `ai/interview_planner.py` | Deterministically ranks topics/stages and asks the LLM only for an objective/expected evidence. |
| `ai/question_generator.py` | Converts a fixed `QuestionPlan` into one natural-language question; planner metadata remains fixed. |
| `ai/answer_assessor.py` | Scores one answer against one question and verifies answer excerpts are verbatim from that transcript. |
| `ai/interview_engine.py` | Owns one stateful interview session and coordinates planner, generator, assessor, coverage, and completion. |
| `ai/report_builder.py` | Derives all numerical results deterministically; uses the LLM only for evidence-cited qualitative feedback. |
| `interview_app_adapter.py` | Session-local application wrapper around matching, engine, and reporting. |
| `application_controller.py` | Dependency-injected UI-event adapter: setup, transcript submission, recovery, and early termination. |
| `myapp.py` | Gradio/PDF/gTTS/Faster Whisper/watsonx setup only. |

## Primary schemas

All AI schemas reject unspecified fields. `Evidence` is an excerpt string; `AnswerEvidence` is explicitly limited to `source="answer"`.

### CandidateProfile

`CandidateProfile` represents resume-supported facts only:

- `candidate_name`, `headline`, and `years_experience_estimate`, each with corresponding evidence when present.
- `skills`, `technologies_tools`.
- `experiences` (organization, role, duration, highlights, skills, achievements, evidence).
- `projects` (name, description, technologies, achievements, evidence).
- `achievements`, `certifications`, and `education`.

Missing information is `null` or `[]`; demographic facts are not inferred.

### RoleProfile

`RoleProfile` contains job-description-supported information:

- `job_title`, `seniority`, `location`, `employment_type`.
- Separate `required_competencies` and `preferred_competencies`; each `Competency` has an id, name, requirements, positive weight, and evidence.
- `technologies_tools` with a required/preferred level, responsibilities, and interview focus.

A competency cannot be both required and preferred.

### SkillMatchReport

One `SkillMatch` is produced for each competency. It carries the competency id/name/weight/requirement level, `strong`, `partial`, or `not_demonstrated` status, candidate evidence, job evidence, and concise reasoning. Positive matches require candidate evidence; `not_demonstrated` requires none and means only “not demonstrated in supplied material.”

`SkillMatchReport` contains required/preferred result lists, level and overall match scores, status ID lists, strongest/weakest areas, and recommended interview focus. It validates no duplicate competency IDs and preserves level separation.

### Interview planning and questions

- `QuestionPlan`: planner-owned `question_id`, optional `competency_id`, stage, question type, difficulty 1–5, objective, expected evidence, optional follow-up reference, and reason.
- `InterviewQuestion`: natural language `question_text` plus metadata copied exactly from `QuestionPlan`.
- `InterviewPlanState`: target count, generated count, current stage, per-competency coverage, follow-up limits/count, and unique plan history.
- `InterviewCoverage`: questions/follow-ups asked, latest and average score, and `not_covered`, `partial`, or `covered` status.

Stages are introduction, resume, technical, behavioral, scenario, and closing. Question types are open ended, technical, behavioral, project deep dive, scenario, and follow-up.

### AnswerAssessment

One assessment has a fixed question and optional competency identity, `overall_score` 0–5, four 0–5 dimensions (`technical_accuracy`, `relevance`, `specificity`, `communication`), verbatim answer evidence, strengths, gaps, action (`follow_up`, `advance`, `change_topic`), optional follow-up focus, and confidence 0–1.

### InterviewSessionState

The engine owns `InterviewSessionState`: session ID, active/completed status, candidate/role/match inputs, planner state, current question, generated questions, exact transcript/assessment history, pending answer signal, and remaining questions. It validates that question/state counts and current question membership remain consistent.

### FinalInterviewReport

The final report is `completed` or `incomplete` and includes:

- deterministic `overall_score`, `job_fit_score`, and recommendation;
- per-assessed-competency results (score, counts, coverage, evidence, strengths/weaknesses/actions);
- four dimension aggregates;
- qualitative strongest/weakest areas, technical and communication feedback, improvement recommendations, and next steps;
- deterministic statistics: question/answer/follow-up counts and competency coverage counts.

## Evidence grounding

1. Profile evidence is checked against the resume or job-description source text using normalized comparison.
2. Skill matching accepts only evidence present in the candidate inventory and selected competency evidence.
3. Question generation receives only selected profile/job context and cannot alter plan metadata.
4. Assessment evidence must occur verbatim in the exact answer transcript. Resume evidence is context only, never proof of an answer.
5. Report evidence is copied from validated assessment evidence and rechecked against original stored transcripts. Qualitative findings must cite that evidence.

## Adaptive logic

The planner ranks required items before preferred items, then considers resume-match status, competency weight, interview coverage, answer gaps, remaining slots, and follow-up limits. Required not-demonstrated and partial competencies receive priority. A strong resume match still requires interview coverage. Low scores with stated gaps can produce follow-ups, bounded by `max_follow_ups`; covered competencies are deprioritized when alternatives exist. The planner reserves an introduction and, when applicable, a closing slot.

## Deterministic scoring

### Skill matching

Strong, partial, and not-demonstrated map to 1.0, 0.5, and 0.0. Each level score is its weighted mean. Overall match is `0.8 × required + 0.2 × preferred` when both exist, otherwise the available level; no competencies produces 0.

### Final report

Dimension percentages are average 0–5 scores divided by 5. Overall interview performance is:

`35% technical accuracy + 25% relevance + 20% specificity + 20% communication`.

Each competency interview score is the average assessment overall score divided by 5. Job fit is the weighted competency-performance score with `80% required + 20% preferred` when both levels were assessed, or the available level otherwise. Resume match is context and never replaces interview performance.

Recommendation bands are strong match ≥90, good match ≥75, developing match ≥60, otherwise weak match. Materially weak required performance caps the recommendation so preferred performance cannot hide it.

## Errors and recovery

- Structured LLM output is parsed/validated and repaired once by `WatsonxStructuredClient`; persistent invalid output raises an error.
- Profile/matching/initialization errors produce no application session.
- Empty answers are assessed deterministically as weak; transcription errors are not submitted.
- Assessment failure leaves the current engine question intact.
- If an assessment was committed but next-question rendering fails, `InterviewApplicationSession.needs_question_retry` is set and `retry_next_question()` avoids reassessing the prior answer.
- If final reporting fails after completion, `build_final_report()` can be retried from unchanged engine state.
- `end_early()` returns an explicitly incomplete report. It is an application-session flag, not an `InterviewEngine` completion transition; do not bypass the adapter to submit another answer afterward.
