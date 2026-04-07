"""
utils/summarizer.py
===================
Produces a plain-English summary of insurance document text.

Strategy (priority order):
  1. OpenAI GPT   – if OPENAI_API_KEY is set in env. Best quality.
  2. Rule-based   – sentence-scoring heuristic (no external API needed).

The rule-based fallback is good enough for demo / offline use.
"""

from __future__ import annotations
import os
import re
from typing import Optional

# ── OpenAI (optional) ─────────────────────────────────────────────────────────
try:
    import openai
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def summarize_text(text: str, max_words: int = 200) -> str:
    """
    Summarize an insurance document.

    Parameters
    ----------
    text      : full document text
    max_words : soft cap on summary length (rule-based path only)

    Returns
    -------
    A summary string.
    """
    if not text or not text.strip():
        return "⚠️ No text found to summarize."

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()

    if _HAS_OPENAI and api_key:
        try:
            return _summarize_with_openai(text, api_key)
        except Exception as exc:
            # Fallback gracefully
            return (
                _summarize_rule_based(text, max_words)
                + f"\n\n*(OpenAI unavailable: {exc})*"
            )

    return _summarize_rule_based(text, max_words)


# ══════════════════════════════════════════════════════════════════════════════
# OPENAI PATH
# ══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """You are an insurance document analyst. Your job is to read insurance 
documents and produce a clear, jargon-free summary for a non-expert customer. 

Your summary should cover:
- What type of document this is (policy, claim form, renewal notice, etc.)
- Who is the insured person / policyholder
- The policy number and coverage type
- Key dates (start, end, renewal)
- Premium amounts and payment frequency
- Main coverage or claim details
- Any important exclusions or conditions

Keep it under 200 words. Use simple, friendly language. Use bullet points for clarity."""


def _summarize_with_openai(text: str, api_key: str) -> str:
    """Call OpenAI chat completion to summarise."""
    client = openai.OpenAI(api_key=api_key)

    # Truncate very long documents to stay within token limits
    truncated = text[:12_000]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": f"Please summarise this insurance document:\n\n{truncated}"},
        ],
        temperature=0.3,
        max_tokens=400,
    )
    return response.choices[0].message.content.strip()


# ══════════════════════════════════════════════════════════════════════════════
# RULE-BASED FALLBACK
# ══════════════════════════════════════════════════════════════════════════════

# Keywords that indicate important sentences
_IMPORTANT_KEYWORDS = [
    "policy", "premium", "coverage", "insured", "claim",
    "expire", "expiry", "renewal", "effective", "amount",
    "sum assured", "sum insured", "deductible", "beneficiary",
    "exclusion", "liable", "coverage period", "plan",
]

# Penalty words (boilerplate / legal filler)
_FILLER_KEYWORDS = [
    "whereas", "hereinafter", "pursuant", "notwithstanding",
    "thereof", "hereof", "aforementioned",
]


def _summarize_rule_based(text: str, max_words: int = 200) -> str:
    """
    Score each sentence by keyword presence and return the top-N sentences
    as a bullet-point summary.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return "Could not generate a summary."

    scored: list[tuple[float, str]] = []
    for sent in sentences:
        sent_lower = sent.lower()
        score = 0.0

        for kw in _IMPORTANT_KEYWORDS:
            if kw in sent_lower:
                score += 1.0

        for kw in _FILLER_KEYWORDS:
            if kw in sent_lower:
                score -= 0.5

        # Prefer sentences with currency symbols (likely key amounts)
        if re.search(r"[\$£€₹]", sent):
            score += 1.5

        # Prefer sentences with years / dates
        if re.search(r"\b(20\d{2}|19\d{2})\b", sent):
            score += 0.5

        # Penalise very short or very long sentences
        word_count = len(sent.split())
        if word_count < 5 or word_count > 60:
            score -= 1.0

        scored.append((score, sent))

    # Sort by score descending, pick top sentences
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [s for _, s in scored if len(s.split()) > 5][:8]

    if not top:
        return "No meaningful content could be extracted for a summary."

    # Re-order by original position (preserves reading order)
    ordered = []
    for s in top:
        pos = text.find(s)
        if pos != -1:
            ordered.append((pos, s))

    ordered = [s for _, s in sorted(ordered)]

    # Format as bullet points
    bullets = "\n".join(f"• {s.strip()}" for s in ordered)
    note = "\n\n*(Summary generated using rule-based extraction — add an OpenAI API key for AI-powered summaries.)*"
    return bullets + note


def _split_sentences(text: str) -> list[str]:
    """Simple sentence splitter using punctuation boundaries."""
    # Normalise whitespace first
    text = re.sub(r"\s+", " ", text)
    # Split on . ! ? followed by space and capital letter
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", text)
    return [s.strip() for s in raw if s.strip()]
