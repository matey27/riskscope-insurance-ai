# 🧠 RiskScope: Insurance Claims Severity & Risk Intelligence System

An end-to-end AI-powered system that transforms unstructured insurance documents into actionable insights using NLP, risk analytics, and LLM-based intelligence.

---

## 🚀 Key Features

- 📄 Upload and process insurance policies and claim documents (PDF)
- 🔍 Intelligent entity extraction (Policy Number, Name, Premium, Dates, Coverage)
- 🧠 Hybrid summarization (Rule-based + LLM-powered explanations)
- ⚠️ Risk analysis engine to detect missing fields and anomalies
- 📊 Claim severity prediction using contextual understanding (RAG-based)
- 🔗 Document similarity comparison for duplicate/fraud detection
- 💬 Chat with documents using LLM-based question answering

---

## 💡 Real-World Impact

Insurance workflows heavily rely on manual document analysis, leading to inefficiencies and errors.

RiskScope addresses this by:
- Reducing document review time from minutes to seconds
- Automating structured data extraction from unstructured PDFs
- Assisting brokers and analysts in faster decision-making
- Enabling scalable risk assessment across large volumes of claims

---

## 🛠️ Tech Stack

- Python
- Streamlit (UI & deployment)
- spaCy (NLP & NER)
- Regex (structured data extraction)
- pdfplumber / PyPDF2 (PDF parsing)
- OpenAI API (LLM-based summarization & Q&A)
- Pandas (data processing)

---

## 🧠 System Architecture

1. PDF Upload → Text extraction  
2. NLP Pipeline → Entity extraction (Regex + spaCy)  
3. Risk Engine → Missing field detection & anomaly analysis  
4. RAG Module → Context-aware claim severity prediction  
5. Similarity Engine → Document comparison  
6. LLM Layer → Summarization & Q&A  

---

## ⚙️ Setup Instructions

```bash
git clone https://github.com/matey27/riskscope-insurance-ai.git
cd riskscope-insurance-ai

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
