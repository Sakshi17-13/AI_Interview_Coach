# AI_Interview_Coach
An **LLM-powered, multimodal AI interview coach** that analyzes resumes and job descriptions to conduct personalized mock interviews using **speech-to-text and text-to-speech technologies**.


---

## 📌 Project Overview

Interview preparation is one of the most important steps in landing a job, but it can often feel stressful and uncertain. Most traditional mock interview tools are generic and do not adapt to a candidate’s background or the specific job role.

**AI Interview Coach** solves this problem by providing a personalized AI-driven interview experience. The system reads the candidate's resume, analyzes the job description, and dynamically generates interview questions tailored to the candidate’s experience and the target role.

The platform also supports **voice interaction**, allowing users to listen to interview questions and respond verbally, creating a realistic interview simulation environment.

This project demonstrates how **Large Language Models (LLMs), speech processing, and web interfaces** can be integrated to build intelligent career preparation tools.

---

## ✨ Key Features

- 📄 **Resume Analysis** – Extracts and summarizes candidate information from PDF resumes  
- 📋 **Job Description Understanding** – Identifies key requirements and expectations for the role  
- 🤖 **Adaptive Interview Questions** – Generates dynamic interview questions based on candidate responses  
- 🔁 **Follow-up Questioning** – Adjusts questions based on previous answers  
- 🔊 **Text-to-Speech** – Converts interview questions into natural audio using gTTS  
- 🎙️ **Speech-to-Text** – Transcribes candidate responses using Faster Whisper  
- 📊 **Performance Evaluation** – Provides AI-generated feedback on interview performance  
- 🌐 **Interactive Web Interface** – Built using Gradio for real-time interaction

---

## 🧠 System Architecture

The system is designed as a **multi-agent AI pipeline** consisting of the following components:

### 1️⃣ Resume Analyst Agent
Extracts and summarizes key information from the candidate's resume.

### 2️⃣ Job Description Expert Agent
Analyzes the job description to identify required skills and expectations.

### 3️⃣ Interview Strategy Agent
Determines the next question strategy based on previous answers.

### 4️⃣ Interviewer Agent
Generates realistic interview questions tailored to the candidate profile.

### 5️⃣ Evaluation Agent
Evaluates candidate responses and provides feedback.

### 6️⃣ Speech Processing Modules
- **Speech-to-Text:** Faster Whisper  
- **Text-to-Speech:** gTTS

### 7️⃣ Web Interface
A **Gradio-based interactive dashboard** that integrates all modules.

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
2.Personalized mock interviews tailored to specific roles
3.AI-assisted communication practice
4.Demonstration of multimodal AI systems combining LLMs and speech technologies

## 📚 Educational Note
This project is developed for academic and learning purposes.
Large Language Models are controlled using structured prompt engineering, which acts as an instruction layer guiding the behavior of the AI system.

The project focuses on AI system integration, conversational workflows, and practical applications of generative AI, rather than training new models.

## 🔮 Future Enhancements
1.Confidence and sentiment analysis of answers
2.Role-specific interview modes (Data Analyst, SWE, ML Engineer, etc.)
3.Interview scoring dashboard
4.Video-based interview simulation
5.Multi-language interview support

## 👩‍💻 Author
**Sakshi Khedkar**
3rd Year Engineering Student
