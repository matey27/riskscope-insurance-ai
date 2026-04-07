"""
Insurance Document Intelligence System
=======================================
Main Streamlit Application Entry Point

This file wires together all utility modules and renders the UI.
Run with: streamlit run app.py
"""

import streamlit as st
from dotenv import load_dotenv
import os

# ── Utility modules ────────────────────────────────────────────────────────────
from utils.pdf_extractor   import extract_text_from_pdf
from utils.entity_extractor import extract_entities
from utils.summarizer      import summarize_text
from utils.risk_analyzer   import analyze_risks
from utils.similarity      import compute_similarity
from utils.qa_module       import answer_question

load_dotenv()

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Document Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* ── Background ── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a4e 50%, #0f0c29 100%);
        min-height: 100vh;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.04);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* ── Card ── */
    .doc-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(12px);
    }

    /* ── Section header ── */
    .section-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.45rem;
        color: #e8d5ff;
        letter-spacing: 0.02em;
        margin-bottom: 0.6rem;
    }

    /* ── Entity badge ── */
    .entity-badge {
        display: inline-block;
        background: rgba(139,92,246,0.22);
        border: 1px solid rgba(139,92,246,0.45);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.82rem;
        color: #d4b4ff;
        margin: 3px 3px 3px 0;
    }

    /* ── Risk flags ── */
    .flag-warn  { background: rgba(251,191,36,0.15); border-left: 4px solid #fbbf24;
                  padding: 0.6rem 1rem; border-radius: 6px; color: #fde68a; margin: 6px 0; }
    .flag-error { background: rgba(239,68,68,0.15);  border-left: 4px solid #ef4444;
                  padding: 0.6rem 1rem; border-radius: 6px; color: #fca5a5; margin: 6px 0; }
    .flag-ok    { background: rgba(52,211,153,0.15); border-left: 4px solid #34d399;
                  padding: 0.6rem 1rem; border-radius: 6px; color: #6ee7b7; margin: 6px 0; }

    /* ── Similarity gauge ── */
    .similarity-score {
        font-family: 'DM Serif Display', serif;
        font-size: 3.5rem;
        color: #a78bfa;
        text-align: center;
        line-height: 1;
    }

    /* ── Chat bubbles ── */
    .chat-user { background: rgba(139,92,246,0.25); border-radius: 14px 14px 4px 14px;
                 padding: 0.65rem 1rem; margin: 6px 0; color: #e9d5ff; text-align: right; }
    .chat-bot  { background: rgba(255,255,255,0.07); border-radius: 14px 14px 14px 4px;
                 padding: 0.65rem 1rem; margin: 6px 0; color: #d1d5db; }

    /* ── Misc ── */
    h1, h2, h3 { color: #e8d5ff !important; }
    .stMarkdown p { color: #c4b5fd; }
    hr { border-color: rgba(255,255,255,0.10) !important; }
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #4f46e5);
        color: white; border: none; border-radius: 10px;
        padding: 0.5rem 1.4rem; font-weight: 600;
        transition: opacity .2s;
    }
    .stButton > button:hover { opacity: 0.85; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR  ── configuration & upload
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛡️ InsureIQ")
    st.caption("Insurance Document Intelligence")
    st.divider()

    st.markdown("### ⚙️ Configuration")
    openai_key = st.text_input(
        "OpenAI API Key (optional)",
        type="password",
        help="Leave blank to use rule-based summarisation & Q&A.",
    )
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

    st.divider()
    st.markdown("### 📂 Upload Documents")

    uploaded_doc1 = st.file_uploader(
        "Primary Document (PDF)", type=["pdf"], key="doc1"
    )
    uploaded_doc2 = st.file_uploader(
        "Second Document for Comparison (PDF, optional)",
        type=["pdf"],
        key="doc2",
    )

    st.divider()
    st.markdown(
        "<small style='color:#6b7280'>Built with ❤️ using spaCy · pdfplumber · "
        "Streamlit · OpenAI</small>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    "<h1 style='font-family:DM Serif Display,serif;font-size:2.6rem;"
    "text-align:center;margin-bottom:0'>🛡️ Insurance Document Intelligence</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center;color:#9ca3af;font-size:1.05rem;margin-top:4px'>"
    "Upload insurance PDFs · extract entities · detect risks · chat with your document</p>",
    unsafe_allow_html=True,
)
st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# HELPER – render entity table
# ══════════════════════════════════════════════════════════════════════════════
def render_entities(entities: dict):
    if not entities:
        st.info("No entities could be extracted.")
        return
    for field, value in entities.items():
        col_a, col_b = st.columns([1, 2])
        col_a.markdown(
            f"<span class='entity-badge'>{field}</span>", unsafe_allow_html=True
        )
        if isinstance(value, list):
            col_b.markdown(", ".join(value) if value else "—")
        else:
            col_b.markdown(str(value) if value else "—")


# ══════════════════════════════════════════════════════════════════════════════
# HELPER – render risk flags
# ══════════════════════════════════════════════════════════════════════════════
def render_flags(flags: list):
    if not flags:
        st.markdown(
            "<div class='flag-ok'>✅ No risk flags detected. Document looks complete.</div>",
            unsafe_allow_html=True,
        )
        return
    for flag in flags:
        severity = flag.get("severity", "warn")
        message  = flag.get("message", str(flag))
        css_cls  = "flag-error" if severity == "error" else "flag-warn"
        icon     = "🔴" if severity == "error" else "⚠️"
        st.markdown(
            f"<div class='{css_cls}'>{icon} {message}</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOGIC  – process document 1
# ══════════════════════════════════════════════════════════════════════════════
if uploaded_doc1 is None:
    st.markdown(
        "<div class='doc-card' style='text-align:center;padding:3rem;'>"
        "<span style='font-size:3rem'>📄</span><br>"
        "<span style='color:#9ca3af'>Upload a PDF in the sidebar to begin</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()


# ── Extract text ───────────────────────────────────────────────────────────────
with st.spinner("📖 Reading document …"):
    text1, extraction_error1 = extract_text_from_pdf(uploaded_doc1)

if extraction_error1:
    st.error(f"Could not read the PDF: {extraction_error1}")
    st.stop()

if not text1.strip():
    st.error("The uploaded PDF appears to be empty or image-only (no selectable text).")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_info, tab_summary, tab_risk, tab_similarity, tab_chat = st.tabs(
    ["📋 Info & Entities", "📝 Summary", "⚠️ Risk Flags", "🔗 Similarity", "💬 Chat"]
)


# ─── TAB 1 : Info & Entities ──────────────────────────────────────────────────
with tab_info:
    st.markdown("<p class='section-title'>Extracted Information</p>", unsafe_allow_html=True)

    with st.spinner("🔍 Extracting entities …"):
        entities = extract_entities(text1)

    st.markdown("<div class='doc-card'>", unsafe_allow_html=True)
    render_entities(entities)
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("📄 View raw extracted text"):
        st.text_area("Raw Text", text1[:5000] + ("…" if len(text1) > 5000 else ""),
                     height=300, disabled=True)


# ─── TAB 2 : Summary ──────────────────────────────────────────────────────────
with tab_summary:
    st.markdown("<p class='section-title'>Document Summary</p>", unsafe_allow_html=True)

    with st.spinner("✍️ Summarising …"):
        summary = summarize_text(text1)

    st.markdown(
        f"<div class='doc-card'><p style='color:#e2e8f0;line-height:1.75'>{summary}</p></div>",
        unsafe_allow_html=True,
    )


# ─── TAB 3 : Risk Flags ───────────────────────────────────────────────────────
with tab_risk:
    st.markdown("<p class='section-title'>Risk & Completeness Analysis</p>", unsafe_allow_html=True)

    with st.spinner("🔎 Analysing document …"):
        flags = analyze_risks(text1, entities)

    render_flags(flags)


# ─── TAB 4 : Similarity ───────────────────────────────────────────────────────
with tab_similarity:
    st.markdown("<p class='section-title'>Document Similarity</p>", unsafe_allow_html=True)

    if uploaded_doc2 is None:
        st.info("Upload a second document in the sidebar to enable comparison.")
    else:
        with st.spinner("📐 Comparing documents …"):
            text2, err2 = extract_text_from_pdf(uploaded_doc2)
            if err2:
                st.error(f"Could not read Document 2: {err2}")
            elif not text2.strip():
                st.error("Document 2 appears empty.")
            else:
                score, explanation = compute_similarity(text1, text2)

                col_l, col_m, col_r = st.columns([1, 1, 1])
                with col_m:
                    st.markdown(
                        f"<div class='similarity-score'>{score}%</div>"
                        "<p style='text-align:center;color:#9ca3af;margin-top:6px'>Similarity Score</p>",
                        unsafe_allow_html=True,
                    )
                st.progress(score / 100)

                st.markdown("<div class='doc-card'>", unsafe_allow_html=True)
                st.markdown(f"**Interpretation:** {explanation}")
                st.markdown("</div>", unsafe_allow_html=True)


# ─── TAB 5 : Chat ─────────────────────────────────────────────────────────────
with tab_chat:
    st.markdown("<p class='section-title'>Chat with Your Document</p>", unsafe_allow_html=True)

    # Initialise session history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display history
    for turn in st.session_state.chat_history:
        role, msg = turn
        css = "chat-user" if role == "user" else "chat-bot"
        align = "right" if role == "user" else "left"
        st.markdown(
            f"<div class='{css}' style='text-align:{align}'>{msg}</div>",
            unsafe_allow_html=True,
        )

    # Input row
    col_q, col_btn = st.columns([5, 1])
    with col_q:
        user_question = st.text_input(
            "Ask a question about the document",
            placeholder="e.g. What is the policy expiry date?",
            label_visibility="collapsed",
            key="chat_input",
        )
    with col_btn:
        ask_clicked = st.button("Ask", use_container_width=True)

    if ask_clicked and user_question.strip():
        st.session_state.chat_history.append(("user", user_question))

        with st.spinner("🤔 Thinking …"):
            answer = answer_question(user_question, text1)

        st.session_state.chat_history.append(("bot", answer))
        st.rerun()

    if st.button("🗑️ Clear chat"):
        st.session_state.chat_history = []
        st.rerun()
