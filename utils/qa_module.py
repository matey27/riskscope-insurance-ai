"""
utils/qa_module.py
==================
Question-Answering over a single document.

Strategy:
  1. OpenAI GPT (best)  – if OPENAI_API_KEY is in env.
  2. Rule-based search  – keyword matching + sentence retrieval.
                          Works offline, no API key needed.

The rule-based approach:
  • Tokenises the question into keywords.
  • Scores each sentence in the document by keyword overlap.
  • Returns the top 2–3 most relevant sentences as the answer.
  • Handles common insurance questions with specialised regex lookups.
"""

from __future__ import annotations
import os
import re

# ── OpenAI (optional) ─────────────────────────────────────────────────────────
try:
    import openai
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def answer_question(question: str, document_text: str) -> str:
    """
    Answer a user question based solely on the document text.

    Parameters
    ----------
    question       : user's natural-language question
    document_text  : full text of the uploaded document

    Returns
    -------
    Answer string.
    """
    if not document_text.strip():
        return "⚠️ No document text available to answer from."

    if not question.strip():
        return "Please type a question above."

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()

    if _HAS_OPENAI and api_key:
        try:
            return _openai_qa(question, document_text, api_key)
        except Exception as exc:
            # Graceful fallback
            fallback = _rule_based_qa(question, document_text)
            return fallback + f"\n\n*(OpenAI unavailable: {exc})*"

    return _rule_based_qa(question, document_text)


# ══════════════════════════════════════════════════════════════════════════════
# OPENAI PATH
# ══════════════════════════════════════════════════════════════════════════════

_SYSTEM_QA_PROMPT = """You are an expert insurance document assistant.
Answer the user's question ONLY based on the document excerpt provided.
If the answer is not in the document, say clearly: "I could not find that information in the document."
Keep answers concise (2-4 sentences). Quote exact values (dates, amounts) when present."""


def _openai_qa(question: str, text: str, api_key: str) -> str:
    """Query OpenAI with the full (truncated) document as context."""
    client = openai.OpenAI(api_key=api_key)

    # Extract most relevant section first to stay within context window
    relevant = _extract_relevant_section(question, text, max_chars=6_000)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": _SYSTEM_QA_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Document excerpt:\n\n{relevant}\n\n"
                    f"Question: {question}"
                ),
            },
        ],
        temperature=0.1,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


# ══════════════════════════════════════════════════════════════════════════════
# RULE-BASED QA
# ══════════════════════════════════════════════════════════════════════════════

# Map common question intents → regex patterns to search in document
_INTENT_PATTERNS: list[tuple[list[str], re.Pattern]] = [
    (
        ["expire", "expiry", "expiration", "end date", "valid until", "valid till"],
        re.compile(
            r"(?:expiry|expiration|valid\s+(?:till|until|to)|end\s+date|maturity)[^\n.]{0,80}",
            re.IGNORECASE,
        ),
    ),
    (
        ["start", "effective", "commencement", "begin", "inception"],
        re.compile(
            r"(?:effective\s+date|start\s+date|commencement|inception)[^\n.]{0,80}",
            re.IGNORECASE,
        ),
    ),
    (
        ["premium", "payment", "installment", "monthly", "annual"],
        re.compile(
            r"(?:premium|payment\s+amount|annual\s+premium|monthly\s+premium)[^\n.]{0,100}",
            re.IGNORECASE,
        ),
    ),
    (
        ["coverage", "cover", "sum assured", "sum insured", "amount"],
        re.compile(
            r"(?:coverage|sum\s+assured|sum\s+insured|covered\s+amount|total\s+coverage)[^\n.]{0,100}",
            re.IGNORECASE,
        ),
    ),
    (
        ["policy number", "policy no", "policy id"],
        re.compile(
            r"(?:policy\s*(?:number|no\.?|#|id))[^\n.]{0,60}",
            re.IGNORECASE,
        ),
    ),
    (
        ["deductible", "excess", "co-pay"],
        re.compile(
            r"(?:deductible|excess|co-?pay(?:ment)?)[^\n.]{0,100}",
            re.IGNORECASE,
        ),
    ),
    (
        ["exclusion", "excluded", "not covered", "does not cover"],
        re.compile(
            r"(?:exclud|not\s+covered|does\s+not\s+cover|no\s+coverage)[^\n.]{0,150}",
            re.IGNORECASE,
        ),
    ),
    (
        ["claim", "claimant", "claim number"],
        re.compile(
            r"(?:claim\s*(?:number|no\.?|#)?)[^\n.]{0,100}",
            re.IGNORECASE,
        ),
    ),
    (
        ["name", "insured", "policyholder", "holder"],
        re.compile(
            r"(?:insured|policyholder|policy\s+holder|name\s+of\s+insured)[^\n.]{0,80}",
            re.IGNORECASE,
        ),
    ),
]

# Stop words to ignore when matching question tokens
_STOP = {
    "what", "when", "where", "who", "which", "how", "is", "the", "a", "an",
    "this", "that", "does", "do", "did", "will", "my", "your", "their",
    "in", "of", "for", "me", "tell", "show", "give", "find", "get",
}


def _rule_based_qa(question: str, text: str) -> str:
    """
    1. Try intent matching with regex patterns.
    2. Fall back to sentence-level keyword search.
    """
    q_lower = question.lower()

    # ── Attempt intent-based lookup ──────────────────────────────────────────
    for keywords, pattern in _INTENT_PATTERNS:
        if any(kw in q_lower for kw in keywords):
            matches = pattern.findall(text)
            if matches:
                snippets = [m.strip() for m in matches[:2]]
                answer = " … ".join(snippets)
                return (
                    f"{answer}\n\n"
                    "*(Answer extracted using keyword search. "
                    "Add an OpenAI API key for smarter responses.)*"
                )

    # ── Fallback: sentence scoring ────────────────────────────────────────────
    question_tokens = {
        w for w in re.findall(r"\b[a-zA-Z]{3,}\b", q_lower)
        if w not in _STOP
    }
    if not question_tokens:
        return "I couldn't understand the question. Please try rephrasing."

    sentences = _split_sentences(text)
    scored = []
    for sent in sentences:
        sent_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", sent.lower()))
        overlap = len(question_tokens & sent_tokens)
        if overlap > 0:
            scored.append((overlap, sent))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [s for _, s in scored[:3]]

    if not top:
        return (
            "I could not find relevant information in the document for your question.\n\n"
            "*(Tip: Add an OpenAI API key for AI-powered answers.)*"
        )

    return (
        " ".join(top)
        + "\n\n*(Answer extracted using keyword search. "
        "Add an OpenAI API key for smarter responses.)*"
    )


def _extract_relevant_section(question: str, text: str, max_chars: int = 6000) -> str:
    """
    For OpenAI: find the most relevant portion of the document using
    sentence scoring, then return up to max_chars of that context.
    """
    if len(text) <= max_chars:
        return text

    q_tokens = {
        w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", question)
        if w.lower() not in _STOP
    }

    # Score paragraphs (cheaper than sentences for long docs)
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    scored = []
    for para in paras:
        tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", para.lower()))
        overlap = len(q_tokens & tokens)
        scored.append((overlap, para))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Build context from top paragraphs up to max_chars
    context = ""
    for _, para in scored:
        if len(context) + len(para) > max_chars:
            break
        context += para + "\n\n"

    return context.strip() or text[:max_chars]


def _split_sentences(text: str) -> list[str]:
    """Simple regex sentence splitter."""
    text = re.sub(r"\s+", " ", text)
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", text)
    return [s.strip() for s in raw if len(s.strip()) > 15]
