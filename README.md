# 🤖 AI Health Chatbot

An intelligent, RAG-powered chatbot designed to interpret health sensor readings and provide medically-informed responses in real time.

This chatbot was developed as part of my **graduation project** and contributed to achieving an **A+ grade** in the Electronics & Communication Engineering program at **Helwan University**.

---
## 🌐 Live Demo

You can try the chatbot online here:

🔗 [Click to Launch the AI Health Chatbot](https://stemchatbotp11.streamlit.app/)

## 🧠 Overview

The AI Health Chatbot leverages **Retrieval-Augmented Generation (RAG)** and Large Language Models to help users understand their medical readings such as ECG, EMG, SpO₂, glucose, and more. It uses an **embedded internal PDF** containing medically verified data to provide fast, accurate, and reliable answers about health metrics, symptoms, and disease indicators.

---

## 🔍 Key Features

- 💬 Chatbot interface with real-time Q&A  
- 🧾 Internal knowledge base from embedded PDF  
- ⚡️ Powered by **LLaMA 3** via **Groq API**  
- 🔍 Uses **BAAI/bge-base-en-v1.5** for text embeddings  
- 📚 Handles user queries related to vital signs and health readings  
- 🕒 Maintains chat history for context-aware conversation  

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit** – user interface
- **LangChain** – RAG framework
- **Groq API** – access to LLaMA 3 LLM
- **HuggingFace Hub** – embedding model access
- **FAISS** – vector database for fast document search
- **PDFTextLoader** – to extract content from internal medical PDF

---

## 🚀 How to Run

Follow these steps to get the chatbot up and running locally:

### 1. Clone the repository

git clone https://github.com/Jimmy70707/AI-Health-Chatbot.git
cd AI-Health-Chatbot

### 2. Install dependencies
Make sure you have Python 3.9+ installed, then run:
pip install -r requirements.txt

### 3. Set up API keys
Create a .env file in the root directory with the following content:
GROQ_API_KEY=your_groq_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token

Replace your_groq_api_key and your_huggingface_token with your actual API keys.

### 4. Run the Streamlit app
streamlit run app.py

## 👨‍💻 Author

**Mohamed Gamal AbdulRahim**  
AI/ML Engineer | Passionate about LLMs, NLP, and real-world AI applications  

📍 6 October, Egypt  

[![Email](https://img.shields.io/badge/-Email-red?style=flat-square&logo=gmail&logoColor=white)](mailto:muhammed.gamal00@gmail.com)
[![LinkedIn](https://img.shields.io/badge/-LinkedIn-blue?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/muhammed-gamal-b0a347244)
[![GitHub](https://img.shields.io/badge/-GitHub-black?style=flat-square&logo=github&logoColor=white)](https://github.com/Jimmy70707)
