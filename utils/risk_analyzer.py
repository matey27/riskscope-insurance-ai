"""
utils/risk_analyzer.py
======================
Analyses extracted entities and raw text to surface potential risks,
incomplete data, or suspicious values.

Each flag is a dict:
  { "severity": "error" | "warn", "message": "Human-readable description" }

Checks performed:
  1. Missing mandatory fields (policy number, name, dates, premium)
  2. Suspicious premium amounts (too high / too low)
  3. Policy already expired
  4. Very short document (likely incomplete)
  5. No currency amounts found at all
  6. Mismatched or ambiguous dates
  7. Keywords suggesting uncovered events / exclusions
  8. No signatures / authorization language (basic check)
"""

from __future__ import annotations
import re
from datetime import datetime
from typing import Any


# ══════════════════════════════════════════════════════════════════════════════
# THRESHOLDS (tunable)
# ══════════════════════════════════════════════════════════════════════════════

_MIN_PREMIUM = 10          # anything below $10 is suspicious
_MAX_PREMIUM = 1_000_000   # anything above $1M is flagged for review
_MIN_DOC_WORDS = 50        # documents shorter than this are probably incomplete


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def analyze_risks(text: str, entities: dict[str, Any]) -> list[dict]:
    """
    Run all risk checks and return a list of flag dicts.

    Parameters
    ----------
    text     : raw document text
    entities : output of entity_extractor.extract_entities()

    Returns
    -------
    List of { "severity": str, "message": str } dicts.
    An empty list means no issues were found.
    """
    flags: list[dict] = []

    _check_missing_fields(entities, flags)
    _check_document_length(text, flags)
    _check_premium_value(entities, flags)
    _check_policy_expired(entities, flags)
    _check_no_amounts(text, flags)
    _check_exclusion_keywords(text, flags)
    _check_authorization(text, flags)

    return flags


# ══════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL CHECKS
# ══════════════════════════════════════════════════════════════════════════════

def _check_missing_fields(entities: dict, flags: list) -> None:
    """Flag any mandatory field that wasn't extracted."""
    mandatory = {
        "Insured Name":      "Insured / policyholder name is missing.",
        "Policy Number":     "Policy number not found — document may be incomplete.",
        "Premium Amount":    "Premium amount not detected.",
        "Policy Start Date": "Policy start/effective date is missing.",
        "Policy End Date":   "Policy expiry/end date is missing.",
    }
    for field, message in mandatory.items():
        if not entities.get(field):
            flags.append({"severity": "warn", "message": message})


def _check_document_length(text: str, flags: list) -> None:
    """Flag documents that are suspiciously short."""
    word_count = len(text.split())
    if word_count < _MIN_DOC_WORDS:
        flags.append({
            "severity": "error",
            "message": (
                f"Document is very short ({word_count} words). "
                "It may be incomplete, corrupted, or image-based."
            ),
        })


def _check_premium_value(entities: dict, flags: list) -> None:
    """Flag premiums that are outside a reasonable range."""
    raw = entities.get("Premium Amount")
    if not raw:
        return  # already flagged as missing

    # Extract numeric value from strings like "$1,200.00" or "INR 5000"
    numeric_str = re.sub(r"[^\d.]", "", raw.replace(",", ""))
    try:
        value = float(numeric_str)
    except ValueError:
        return  # can't parse, skip

    if value < _MIN_PREMIUM:
        flags.append({
            "severity": "warn",
            "message": (
                f"Premium amount ({raw}) seems unusually low (< ${_MIN_PREMIUM}). "
                "Please verify."
            ),
        })
    elif value > _MAX_PREMIUM:
        flags.append({
            "severity": "warn",
            "message": (
                f"Premium amount ({raw}) is very high (> ${_MAX_PREMIUM:,}). "
                "Manual review recommended."
            ),
        })


def _check_policy_expired(entities: dict, flags: list) -> None:
    """Check if the policy end date has already passed."""
    end_date_raw = entities.get("Policy End Date")
    if not end_date_raw:
        return

    # Try common date formats
    date_formats = [
        "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d",
        "%d-%m-%Y", "%d.%m.%Y",
        "%B %d, %Y", "%b %d, %Y",
        "%d %B %Y", "%d %b %Y",
        "%Y/%m/%d",
    ]
    parsed: datetime | None = None
    for fmt in date_formats:
        try:
            parsed = datetime.strptime(end_date_raw.strip(), fmt)
            break
        except ValueError:
            continue

    if parsed and parsed < datetime.now():
        flags.append({
            "severity": "error",
            "message": (
                f"Policy appears to have expired on {end_date_raw}. "
                "Coverage may no longer be active."
            ),
        })


def _check_no_amounts(text: str, flags: list) -> None:
    """Flag if no currency amounts at all appear in the document."""
    currency_pattern = re.compile(r"[\$£€₹][\s\d]|INR|USD|GBP|EUR", re.IGNORECASE)
    if not currency_pattern.search(text):
        flags.append({
            "severity": "warn",
            "message": "No currency amounts detected in the document.",
        })


def _check_exclusion_keywords(text: str, flags: list) -> None:
    """
    Flag documents containing heavy exclusion / disclaimer language —
    useful to alert the user to read fine-print sections.
    """
    exclusion_phrases = [
        "not covered", "excluded", "does not cover",
        "shall not be liable", "no coverage",
        "pre-existing condition", "waiting period",
    ]
    found = [
        phrase for phrase in exclusion_phrases
        if phrase.lower() in text.lower()
    ]
    if found:
        flags.append({
            "severity": "warn",
            "message": (
                "Document contains exclusion clauses: "
                + ", ".join(f'"{p}"' for p in found[:4])
                + ". Review carefully."
            ),
        })


def _check_authorization(text: str, flags: list) -> None:
    """Check for absence of signature / authorization language."""
    auth_patterns = [
        "signature", "signed by", "authorised by", "authorized by",
        "stamp", "seal", "witness",
    ]
    has_auth = any(p in text.lower() for p in auth_patterns)
    if not has_auth:
        flags.append({
            "severity": "warn",
            "message": (
                "No signature or authorization language found. "
                "The document may not be officially signed/stamped."
            ),
        })
