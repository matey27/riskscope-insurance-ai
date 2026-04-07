# 🛡️ Insurance Document Intelligence System

An end-to-end NLP-based application to analyze insurance documents by extracting key entities, detecting risks, summarizing content, and enabling document-based Q&A.

---

## 🚀 Features

- 📄 Upload insurance PDFs
- 🔍 Entity Extraction (Policy No, Name, Premium, Dates)
- 🧠 Text Summarization (Rule-based + LLM)
- ⚠️ Risk Detection (Missing fields, anomalies)
- 🔗 Document Similarity Comparison
- 💬 Chat with Document (LLM-powered)

---

## 🛠️ Tech Stack

- Python
- Streamlit
- spaCy (NLP)
- pdfplumber (PDF extraction)
- OpenAI API (LLM)

---

## 🧠 Architecture

1. PDF → Text extraction  
2. NLP pipeline → Entity extraction  
3. Rule-based + LLM → Summary  
4. Risk engine → Flags anomalies  
5. Similarity module → Compare docs  
6. Chat module → Query documents  

---

## ⚙️ Setup Instructions

```bash
git clone https://github.com/matey27/insurance-document-intelligence.git
cd insurance-document-intelligence

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
