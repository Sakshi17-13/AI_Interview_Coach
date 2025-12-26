# AI_Interview_Coach
An LLM-powered, multimodal AI interview coach that analyzes resumes and job descriptions to conduct personalized mock interviews using speech-to-text and text-to-speech

---

## 📌 Project Overview

Interview preparation is a critical but often stressful step in landing a job. Most traditional mock interview tools are generic and do not adapt to a candidate’s background or the specific job role.

The **AI Interview Coach** is an intelligent system that:
- Reads and understands a candidate’s resume
- Analyzes the job description for required skills and expectations
- Conducts a personalized, adaptive mock interview
- Interacts using voice (audio questions and spoken answers)
- Evaluates the candidate’s performance and provides feedback

This project demonstrates how Large Language Models (LLMs) and speech technologies can be integrated into a real-world, user-facing AI application.

---

## ✨ Key Features

- 📄 Resume PDF parsing and summarization  
- 📋 Job description analysis using LLM reasoning  
- 🤖 Personalized interview question generation  
- 🔁 Adaptive follow-up questioning based on previous answers  
- 🔊 Text-to-Speech interviewer voice (gTTS)  
- 🎙️ Speech-to-Text transcription using Faster Whisper  
- 📊 Automated interview performance evaluation  
- 🌐 Interactive Gradio-based web interface  

---

## 🧠 System Architecture

The system is designed as a **multi-agent AI pipeline**:

1. **Resume Analyst Agent**  
   Extracts and summarizes key information from the resume PDF.

2. **Job Description Expert Agent**  
   Identifies required skills, responsibilities, and expectations from the job description.

3. **Interview Strategy Agent**  
   Decides whether to ask a new topic or a follow-up question based on interview history.

4. **Interviewer Agent**  
   Generates natural, role-specific interview questions.

5. **Evaluation Agent**  
   Assesses the candidate’s responses and provides feedback on skills and personality fit.

6. **Speech-to-Text Module**  
   Converts user’s spoken answers into text using Faster Whisper.

7. **Text-to-Speech Module**  
   Converts interviewer questions into natural-sounding audio.

8. **Gradio Web Interface**  
   Integrates all components into an interactive user experience.

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **LLM Platform:** IBM watsonx.ai (LLaMA 3 Instruct Model)  
- **Speech-to-Text:** Faster Whisper  
- **Text-to-Speech:** gTTS  
- **PDF Parsing:** PyPDF2  
- **Web Interface:** Gradio  

---

## 🚀 How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ai-interview-coach.git
   cd ai-interview-coach
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the application:
   ```bash
   python app.py
4. Open the generated Gradio link in your web browser.

## 🎯 Use Cases
1.Interview preparation for students and job seekers
2.Personalized mock interviews for specific job roles
3.AI-assisted communication and confidence practice
4.Demonstration of LLM-based multimodal systems

## 📚 Educational Note
This project is developed for academic and learning purposes.
Large Language Models are controlled using structured prompt design, which acts as a rule-based instruction layer to guide intelligent behavior. The focus of the project is on system integration, workflow design, and practical AI application, not on model training.

## 🔮 Future Enhancements
1.Confidence and sentiment analysis of answers
2.Role-specific interview modes (Data Analyst, SWE, ML Engineer, etc.)
3.Interview scoring dashboard
4.Video-based interview simulation
5.Multi-language interview support

## 👩‍💻 Author
**Sakshi Khedkar**
3rd Year Engineering Student
