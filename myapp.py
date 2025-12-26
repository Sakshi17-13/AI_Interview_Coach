from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters
from ibm_watsonx_ai import Credentials

from gtts import gTTS
from faster_whisper import WhisperModel

import PyPDF2

import gradio as gr

# Global variable of the system: track the chat histories, number of interview questions, 
# the resume and job summary.
chat_histories = {}
interview_step = 0
resume_summary = None
job_summary = None

project_id="skills-network"

credentials = Credentials(
                url = "https://us-south.ml.cloud.ibm.com"
                )

# Get sample parameter values
sample_params = TextChatParameters.get_sample_params()
sample_params['max_tokens'] = int(1e5)
sample_params['response_format'] = None

# Initialize the TextChatParameters object with the sample values
params = TextChatParameters(**sample_params)

# Define LLM
llm_base = ModelInference(
    model_id='meta-llama/llama-3-3-70b-instruct',
    credentials=credentials,
    project_id=project_id,
    params=params,
)

def Resume_Analyst(resume):
    prompt = f"""
    Write a detailed REPORT, in several paragraphs, on the candidate. 
    Three paragraphs: candidate's name and demographic info, key_skills, and the summary of the past experiences.

    Resume:
    {resume}
    """

    response = llm_base.chat(
        messages=[
                {"role":"system","content":"You are an HR expert in reviewing resumes."},
                {"role": "user", "content":prompt}
            ]
        )
    response_output = response['choices'][0]['message']['content']
    return response_output

def Job_Description_Expert(job_description):
    prompt = f"""
    Write a summary of the job description.
    Identify the skills required and the experiences preferred.

    Job Description:
    {job_description}
    """

    response = llm_base.chat(
        messages=[
                {"role":"system","content":"You are job expert."},
                {"role": "user", "content":prompt}
            ]
        )
    response_output = response['choices'][0]['message']['content']
    return response_output

def Interview_Question_Action(chat_histories, resume_summary, job_summary):
    prompt = f"""
    Based on the histories of the answers, the resume summary and the job summary,
    pick one of the following two actions for the next question:
    - (1) Ask about another past experience or skills on the resume.
    - (2) Ask follow-up questions of the current topic.

    Answer Histories:
    {chat_histories}

    Resume Summary:
    {resume_summary}

    Job Summary:
    {job_summary}
    """

    response = llm_base.chat(
        messages=[
                {"role":"system","content":"You are job expert."},
                {"role": "user", "content":prompt}
            ]
        )
    response_output = response['choices'][0]['message']['content']
    return response_output

def Interviewer(resume_summary, job_summary, action=None, last=False):
    if not last:
        if action is not None:
            prompt = f"""
            Directly ask the question based on the given action instruction, 
            resume summary and the job summary.

            DO NOT GIVE ANY EXPLANATIONS WHY YOU ASK THE QUESTION.

            Action:
            {action}

            Resume Summary:
            {resume_summary}

            Job Summary:
            {job_summary}
            """

            response = llm_base.chat(
            messages=[
                    {"role":"system","content":"You are an expert interviewer."},
                    {"role": "user", "content":prompt}
                ]
            )
            response_output = response['choices'][0]['message']['content']
        else:
            response_output = "Tell me about yourself."
    else:
        prompt = f"""
            The interview ends. Wrap up and express gratitude towards the candidate based on the resume.
            Be CONCISE.

            Resume Summary:
            {resume_summary}
            """

        response = llm_base.chat(
        messages=[
                {"role":"system","content":"You are an expert interviewer."},
                {"role": "user", "content":prompt}
            ]
        )
        response_output = response['choices'][0]['message']['content']
    
    return response_output

def Evaluator(chat_histories, job_summary):
    prompt = f"""
    Based on the histories of the answers and the job summary,
    evaluate if the candidate is a good match, by personality and skills, 
    and give reasons.

    Answer Histories:
    {chat_histories}

    Job Summary:
    {job_summary}
    """

    response = llm_base.chat(
        messages=[
                {"role":"system","content":"You are an expert to judge the performance of an interviewee."},
                {"role": "user", "content":prompt}
            ]
        )
    response_output = response['choices'][0]['message']['content']
    return response_output

# Function to read the pdf file (for resume)
def extract_text_from_pdf(pdf_file_path):
    reader = PyPDF2.PdfReader(pdf_file_path.name)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()

# Function to generate the audio file
def text_to_speech_file(text_input):
    # 1. Generate the audio and save it
    audio_file_path = "temp_voice.mp3"
    
    # Using gTTS (as demonstrated earlier) to create the audio file
    tts = gTTS(text=text_input, lang='en')
    tts.save(audio_file_path)
    
    # 2. Return the file path
    # Gradio will automatically display an audio player for this file.
    return audio_file_path

# Function to convert the audio file to text
def transcribe_audio_faster_whisper(
    audio_file_path: str, 
    model_size: str = "base", 
    device: str = "auto",
    compute_type: str = "auto"
) -> str:
    # ... (function body remains the same as previously defined)
    
    if audio_file_path is None:
        return "Please provide an audio input."
        
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
        
    device = "cpu"
        
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        segments, info = model.transcribe(audio_file_path, beam_size=5)
        full_transcript = [segment.text for segment in segments]

        return "".join(full_transcript).strip()

    except Exception as e:
        return f"❌ An error occurred during transcription: {e}"

def next_question(resume_path, job_str, total_number, question_previous="", answer_previous=""):
    # Refer to the global variables defined at the front
    global chat_histories, interview_step, resume_summary, job_summary
    
    # Generate resume_summary if it hasn't been done
    if resume_summary is None:
        resume_summary = extract_text_from_pdf(resume_path)
        
    # Generate job_summary if it hasn't been done
    if job_summary is None:
        job_summary = Job_Description_Expert(job_str)
    
    # Converts the user’s last voice answer into text using the Whisper speech recognition model
    try:
        answer_previous = transcribe_audio_faster_whisper(answer_previous)
    except:
        answer_previous = ""
    
    # Update the interview history after the interview starts. 
    # The chat history is formatted as a dictionary with keys of interview questions 
    # and their values of interviewee's answers.
    if interview_step > 0:
        chat_histories[f"Q{interview_step+1}: {question_previous}"] = answer_previous
    
    # If it’s the first question, it defaults to “Tell me about yourself.”
    # Otherwise, the Interview Question Action Agent decides whether to:
    # Ask about another resume topic, or Ask a deeper follow-up question.
    # The Interviewer Agent then formulates the next question naturally.
    if interview_step < total_number:
        if interview_step == 0:
            action = None
        else:
            chat_hist_str = str(chat_histories)
            action = Interview_Question_Action(chat_hist_str, resume_summary, job_summary)
        
        Question_next = Interviewer(resume_summary, job_summary, action, last=False)
    else:
        Question_next = Interviewer(resume_summary, job_summary, action=None, last=True)
    
    # If the interview ends, the evaluator will evaluate interviewee's performance.
    if interview_step >= total_number:
        evaluation = Evaluator(str(chat_histories), job_summary)
        chat_histories = {}
        interview_step = 0
        resume_summary = None
        job_summary = None
    else:
        evaluation = "Evaluation Ongoing ......"
    
    # Convert the next interview question to the audio.
    question_audio_path = text_to_speech_file(Question_next)
    interview_step += 1

    return gr.update(value=question_audio_path), gr.update(value=None), gr.update(value="Submit!"), gr.update(value=evaluation)

# gradio ui
with gr.Blocks() as demo:
    gr.Markdown("# Personalized Interview Coach")
    gr.Markdown('## Upload your pdf resume/CV and copy paste the job description you are applying to:')
    
    # The first row contains two main inputs side-by-side: resume_input and job_desc_input.
    with gr.Row():
        resume_input = gr.File(label="Upload Resume (PDF)",type='filepath')
        job_desc_input = gr.Textbox(label="Job Description", lines=15)
        
    # Input for the length of the interview
    gr.Markdown('## Decide the length of your mock interview (from 1 question to 10 questions):')
    num_q_input = gr.Slider(label="Number of Questions", minimum=1, maximum=10, value=5, step=1)
    
    # Define the window for interviewer question audio
    gr.Markdown('## Click "Start Interview" below to start the Mock Interview!')
    interviewer_question = gr.Audio(label="Interviewer Question", type="filepath")
    
    # Define the window for interviewee's audio input
    user_answer = gr.Audio(
            sources=["microphone"], # Only allows microphone input
            type="filepath",        # Returns the path to the temporary recorded file
            label="Your turn! Record Your Answer."
        )
        
    # Define the start interview button
    start_btn = gr.Button("Start Interview", scale=2, min_width=200)
    
    # Define the evaluation textbox that shows the final feedback
    gr.Markdown("## Evaluating your performance along the way ...")
    evaluation_textbox = gr.Textbox(label="Performance Evaluation",lines=20)
    
    # Integrate the next_question() function with the start button
    start_btn.click(
        fn=next_question,
        inputs=[resume_input, job_desc_input, num_q_input, interviewer_question, user_answer],
        outputs=[interviewer_question, user_answer, start_btn, evaluation_textbox]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True)
