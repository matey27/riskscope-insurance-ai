"""
utils/entity_extractor.py
=========================
Hybrid entity extraction: Regex patterns first, then spaCy NER as a fallback
or complement.

Extracts:
  - Insured Name / Policy Holder
  - Policy Number
  - Premium Amount
  - Coverage Amount
  - Policy Start Date
  - Policy End / Expiry Date
  - Claim Number (if present)

The hybrid approach is intentional:
  • Regex  – fast, deterministic, great for structured fields (numbers, dates).
  • spaCy  – catches names and dates written in free-form prose.
"""

from __future__ import annotations
import re
from typing import Any

# ── spaCy (optional) ──────────────────────────────────────────────────────────
try:
    import spacy
    # Use the small English model; download with: python -m spacy download en_core_web_sm
    try:
        _NLP = spacy.load("en_core_web_sm")
        _HAS_SPACY = True
    except OSError:
        _NLP = None
        _HAS_SPACY = False
except ImportError:
    _NLP = None
    _HAS_SPACY = False


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def extract_entities(text: str) -> dict[str, Any]:
    """
    Return a dict with all extracted insurance entities.

    Returns
    -------
    {
        "Insured Name"      : str | None,
        "Policy Number"     : str | None,
        "Claim Number"      : str | None,
        "Premium Amount"    : str | None,
        "Coverage Amount"   : str | None,
        "Policy Start Date" : str | None,
        "Policy End Date"   : str | None,
        "Dates Found"       : list[str],   # all dates, for completeness
    }
    """
    entities: dict[str, Any] = {
        "Insured Name":      None,
        "Policy Number":     None,
        "Claim Number":      None,
        "Premium Amount":    None,
        "Coverage Amount":   None,
        "Policy Start Date": None,
        "Policy End Date":   None,
        "Dates Found":       [],
    }

    # ── Step 1: Regex extraction ───────────────────────────────────────────────
    _regex_extract(text, entities)

    # ── Step 2: spaCy NER to fill in blanks ───────────────────────────────────
    if _HAS_SPACY and _NLP is not None:
        _spacy_extract(text, entities)

    return entities


# ══════════════════════════════════════════════════════════════════════════════
# REGEX PATTERNS
# ══════════════════════════════════════════════════════════════════════════════

# ── Policy number ─────────────────────────────────────────────────────────────
_POL_NUM_RE = re.compile(
    r"""
    (?:policy\s*(?:number|no\.?|\#)\s*[:\-]?\s*)   # label
    ([A-Z0-9]{4,20}(?:[-/][A-Z0-9]+)*)            # value
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ── Claim number ──────────────────────────────────────────────────────────────
_CLAIM_NUM_RE = re.compile(
    r"""
    (?:claim\s*(?:number|no\.?|\#|id)\s*[:\-]?\s*)
    ([A-Z0-9]{4,20}(?:[-/][A-Z0-9]+)*)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ── Currency amounts ──────────────────────────────────────────────────────────
_CURRENCY_RE = re.compile(
    r"""
    (?:[\$£€₹])\s*[\d,]+(?:\.\d{2})?     # symbol-first:  $1,000.00
    |
    [\d,]+(?:\.\d{2})?\s*(?:USD|INR|GBP|EUR)  # value-first: 1000 USD
    """,
    re.VERBOSE,
)

# ── Premium context ───────────────────────────────────────────────────────────
_PREMIUM_RE = re.compile(
    r"""
    (?:premium|annual\s+premium|monthly\s+premium)\s*[:\-]?\s*
    ([\$£€₹]?\s*[\d,]+(?:\.\d{2})?)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ── Coverage / sum assured ────────────────────────────────────────────────────
_COVERAGE_RE = re.compile(
    r"""
    (?:coverage\s+amount|sum\s+assured|sum\s+insured|
       coverage|total\s+coverage|insured\s+amount)\s*[:\-]?\s*
    ([\$£€₹]?\s*[\d,]+(?:\.\d{2})?)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ── Insured name ──────────────────────────────────────────────────────────────
_NAME_RE = re.compile(
    r"""
    (?:insured(?:\s+name)?|policy\s*holder(?:\s+name)?|
       name\s+of\s+insured|applicant(?:\s+name)?)\s*[:\-]?\s*
    ([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ── Dates – multiple formats ──────────────────────────────────────────────────
_DATE_RE = re.compile(
    r"""
    \b
    (?:
        \d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}         # 01/01/2024
        |
        \d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2}            # 2024-01-01
        |
        (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}  # Jan 1, 2024
        |
        \d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}    # 1 Jan 2024
    )
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ── Start/end date context ────────────────────────────────────────────────────
_START_DATE_RE = re.compile(
    r"""
    (?:effective\s+date|start\s+date|commencement\s+date|inception\s+date|
       from\s+date|policy\s+start)\s*[:\-]?\s*
    (\S+(?:\s+\S+){0,2})
    """,
    re.IGNORECASE | re.VERBOSE,
)

_END_DATE_RE = re.compile(
    r"""
    (?:expiry\s+date|expiration\s+date|end\s+date|maturity\s+date|
       valid\s+(?:till|until|to)|policy\s+end|renewal\s+date)\s*[:\-]?\s*
    (\S+(?:\s+\S+){0,2})
    """,
    re.IGNORECASE | re.VERBOSE,
)


# ══════════════════════════════════════════════════════════════════════════════
# PRIVATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _regex_extract(text: str, entities: dict) -> None:
    """Fill `entities` dict using regex patterns."""

    # Policy number
    m = _POL_NUM_RE.search(text)
    if m:
        entities["Policy Number"] = m.group(1).strip()

    # Claim number
    m = _CLAIM_NUM_RE.search(text)
    if m:
        entities["Claim Number"] = m.group(1).strip()

    # Premium
    m = _PREMIUM_RE.search(text)
    if m:
        entities["Premium Amount"] = m.group(1).strip()

    # Coverage
    m = _COVERAGE_RE.search(text)
    if m:
        entities["Coverage Amount"] = m.group(1).strip()

    # Insured name
    m = _NAME_RE.search(text)
    if m:
        entities["Insured Name"] = m.group(1).strip()

    # All dates in document
    all_dates = _DATE_RE.findall(text)
    entities["Dates Found"] = list(dict.fromkeys(all_dates))[:10]  # deduplicate, cap at 10

    # Start date
    m = _START_DATE_RE.search(text)
    if m:
        raw = m.group(1).strip()
        # Grab just the date portion
        dm = _DATE_RE.search(raw)
        entities["Policy Start Date"] = dm.group(0) if dm else raw[:30]

    # End date
    m = _END_DATE_RE.search(text)
    if m:
        raw = m.group(1).strip()
        dm = _DATE_RE.search(raw)
        entities["Policy End Date"] = dm.group(0) if dm else raw[:30]


def _spacy_extract(text: str, entities: dict) -> None:
    """Use spaCy NER to fill in fields still missing after regex."""
    # Truncate to avoid memory issues on very large documents
    doc = _NLP(text[:50_000])

    # ── PERSON → Insured Name ──────────────────────────────────────────────────
    if entities["Insured Name"] is None:
        persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        if persons:
            entities["Insured Name"] = persons[0]

    # ── DATE entities → fill start/end if still missing ───────────────────────
    dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
    if dates:
        if not entities["Dates Found"]:
            entities["Dates Found"] = dates[:10]
        if entities["Policy Start Date"] is None and len(dates) >= 1:
            entities["Policy Start Date"] = dates[0]
        if entities["Policy End Date"] is None and len(dates) >= 2:
            entities["Policy End Date"] = dates[-1]

    # ── MONEY entities → fill premium if missing ──────────────────────────────
    if entities["Premium Amount"] is None:
        money = [ent.text for ent in doc.ents if ent.label_ == "MONEY"]
        if money:
            entities["Premium Amount"] = money[0]
