"""
utils/similarity.py
===================
Compares two insurance documents and returns a similarity score (0–100).

Algorithm:
  Primary   – TF-IDF cosine similarity (scikit-learn). Accurate and fast.
  Fallback  – Jaccard similarity on word sets (no dependencies beyond stdlib).

The function also returns a human-readable interpretation of the score.
"""

from __future__ import annotations
import re
from typing import Tuple

# ── scikit-learn (optional but recommended) ────────────────────────────────────
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def compute_similarity(text1: str, text2: str) -> Tuple[int, str]:
    """
    Compute textual similarity between two documents.

    Parameters
    ----------
    text1, text2 : full document texts

    Returns
    -------
    (score, explanation)
      score       – integer 0–100
      explanation – plain-English interpretation
    """
    if not text1.strip() or not text2.strip():
        return 0, "One or both documents are empty — cannot compute similarity."

    if _HAS_SKLEARN:
        score = _tfidf_similarity(text1, text2)
    else:
        score = _jaccard_similarity(text1, text2)

    explanation = _interpret(score)
    return score, explanation


# ══════════════════════════════════════════════════════════════════════════════
# SIMILARITY ALGORITHMS
# ══════════════════════════════════════════════════════════════════════════════

def _tfidf_similarity(text1: str, text2: str) -> int:
    """
    TF-IDF vectorisation + cosine similarity.
    Returns an integer 0–100.
    """
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5_000,         # cap vocabulary for performance
        ngram_range=(1, 2),         # unigrams + bigrams
        sublinear_tf=True,          # dampens high term frequencies
    )
    try:
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        cosine = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(float(cosine) * 100)
    except Exception:
        # If TF-IDF fails (e.g., empty vocabulary), fall back to Jaccard
        return _jaccard_similarity(text1, text2)


def _jaccard_similarity(text1: str, text2: str) -> int:
    """
    Jaccard similarity on lowercased word sets.
    Less precise than TF-IDF but requires no third-party libraries.
    Returns an integer 0–100.
    """
    set1 = _tokenise(text1)
    set2 = _tokenise(text2)

    if not set1 or not set2:
        return 0

    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return round((intersection / union) * 100) if union > 0 else 0


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

# Common English stop words (minimal set for the Jaccard fallback)
_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "is", "it", "be", "as", "by", "this", "that", "with", "from",
    "are", "was", "were", "has", "have", "had", "not", "do", "does", "did",
    "will", "would", "shall", "should", "may", "might", "can", "could",
    "its", "if", "than", "then", "so", "up", "out", "any", "all", "there",
}


def _tokenise(text: str) -> set[str]:
    """Lower-case, strip punctuation, remove stop words, return word set."""
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return {w for w in words if w not in _STOP_WORDS}


def _interpret(score: int) -> str:
    """Return a plain-English interpretation of the similarity score."""
    if score >= 90:
        return (
            "The documents are nearly identical — possibly the same document "
            "with minor edits (e.g., different dates or names)."
        )
    elif score >= 70:
        return (
            "High similarity. Documents likely cover the same policy type, "
            "product family, or insurer template."
        )
    elif score >= 50:
        return (
            "Moderate similarity. Documents share common insurance terminology "
            "but differ significantly in content."
        )
    elif score >= 25:
        return (
            "Low similarity. Documents are mostly different — they may cover "
            "different insurance types or be from different insurers."
        )
    else:
        return (
            "Very low similarity. The documents appear unrelated or cover "
            "entirely different subject matter."
        )
