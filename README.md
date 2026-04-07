# 🛡️ Insurance Document Intelligence System

A production-style NLP web application built with **Streamlit** that lets you upload insurance PDFs and instantly:

- 📋 **Extract** structured entities (name, policy number, premium, dates)
- 📝 **Summarize** documents in plain English
- ⚠️ **Flag risks** (missing fields, expired policy, suspicious amounts)
- 🔗 **Compare** two documents for similarity
- 💬 **Chat** with your document using natural-language questions

---

## 🏗️ Architecture

```
insurance_doc_intelligence/
├── app.py                    ← Streamlit UI (entry point)
├── requirements.txt
├── .env.example
└── utils/
    ├── __init__.py
    ├── pdf_extractor.py      ← PDF → text (pdfplumber + PyPDF2)
    ├── entity_extractor.py   ← Hybrid Regex + spaCy NER
    ├── summarizer.py         ← OpenAI GPT or rule-based
    ├── risk_analyzer.py      ← Rule-based risk flagging
    ├── similarity.py         ← TF-IDF cosine or Jaccard
    └── qa_module.py          ← OpenAI GPT or keyword-search Q&A
```

Each module is self-contained with a single public function. Swap one out without touching the rest.

---

## ⚡ Quick Start

### 1. Clone / download the project

```bash
git clone <your-repo>
cd insurance_doc_intelligence
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the spaCy language model

```bash
python -m spacy download en_core_web_sm
```

### 5. (Optional) Set your OpenAI API key

```bash
cp .env.example .env
# Edit .env and paste your key
```

Or just enter it in the sidebar when the app is running.

### 6. Run the app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🔑 OpenAI API Key — Do I need it?

**No!** The app has intelligent fallbacks for every feature:

| Feature          | Without API Key              | With API Key            |
|------------------|------------------------------|-------------------------|
| PDF Extraction   | ✅ pdfplumber / PyPDF2        | same                    |
| Entity Extraction| ✅ Regex + spaCy NER          | same                    |
| Summarization    | ✅ Sentence scoring heuristic | ✅ GPT-3.5 Turbo         |
| Risk Analysis    | ✅ Rule-based checks          | same                    |
| Similarity       | ✅ TF-IDF cosine              | same                    |
| Chat / Q&A       | ✅ Keyword search             | ✅ GPT-3.5 Turbo         |

---

## 📦 Module Reference

### `utils/pdf_extractor.py`
**`extract_text_from_pdf(uploaded_file) → (text, error)`**  
Tries pdfplumber first, falls back to PyPDF2. Returns `(text, None)` on success or `("", error_message)` on failure.

### `utils/entity_extractor.py`
**`extract_entities(text) → dict`**  
Hybrid approach: regex patterns for structured fields + spaCy NER for names and dates.

### `utils/summarizer.py`
**`summarize_text(text, max_words=200) → str`**  
GPT-3.5-turbo if key is set, else sentence-scoring extraction summary.

### `utils/risk_analyzer.py`
**`analyze_risks(text, entities) → list[dict]`**  
Returns flags: `[{"severity": "error"|"warn", "message": "..."}]`.

### `utils/similarity.py`
**`compute_similarity(text1, text2) → (score_int, interpretation_str)`**  
TF-IDF cosine similarity (0–100) with Jaccard fallback.

### `utils/qa_module.py`
**`answer_question(question, document_text) → str`**  
Intent-matching + sentence scoring fallback. GPT-3.5 if key is set.

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: spacy` | `pip install spacy && python -m spacy download en_core_web_sm` |
| `OSError: [E050] Can't find model 'en_core_web_sm'` | `python -m spacy download en_core_web_sm` |
| PDF returns empty text | PDF is likely image-based (scanned). Use OCR tools first. |
| OpenAI `AuthenticationError` | Check your API key in the sidebar or `.env` |
| Port already in use | `streamlit run app.py --server.port 8502` |

---

## 🚀 Extending the Project

- **Add OCR**: Integrate `pytesseract` + `pdf2image` for scanned PDFs.
- **Better embeddings**: Replace TF-IDF with `sentence-transformers` for semantic similarity.
- **Database**: Store extracted entities in SQLite or PostgreSQL.
- **Authentication**: Add `streamlit-authenticator` for multi-user access.
- **Batch upload**: Process multiple PDFs at once with `asyncio`.
