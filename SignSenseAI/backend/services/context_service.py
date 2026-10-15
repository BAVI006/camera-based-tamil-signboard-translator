"""Evidence-grounded sign categorization and concise explanations."""

import json
from pathlib import Path
import re
import unicodedata
from typing import Any


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CATEGORY_TERMS_PATH = DATA_DIR / "category_terms.json"
VERIFIED_PHRASES_PATH = DATA_DIR / "verified_phrases.json"

CATEGORY_LABELS = {
    "transport": "Transport and public transit",
    "directions_places": "Directions and places",
    "healthcare": "Healthcare",
    "education": "Education",
    "shops_businesses": "Shops and businesses",
    "warnings_prohibitions": "Warnings and prohibitions",
    "government_public_services": "Government and public services",
    "other_unknown": "Other or unknown",
}


def _load_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not load context data from {path}: {exc}") from exc


def classify_sign(
    tamil_text: str,
    english_translation: str | None = None,
    confidence: float | None = None,
) -> dict[str, Any]:
    """Classify by explicit phrase cues and explain only what the evidence supports."""
    if not tamil_text.strip():
        return {
            "category": "other_unknown",
            "context": "No readable sign text was detected, so its practical meaning cannot be determined.",
            "needs_review": True,
            "evidence": [],
        }

    verified_phrase = lookup_verified_phrase(tamil_text)
    if verified_phrase and verified_phrase.get("context"):
        return {
            "category": verified_phrase["category"],
            "context": verified_phrase["context"],
            "needs_review": confidence is None or confidence < 50,
            "evidence": [verified_phrase["tamil"]],
        }

    combined = f"{tamil_text} {english_translation or ''}".casefold()
    terms = _load_json(CATEGORY_TERMS_PATH)
    scores: dict[str, int] = {}
    evidence: dict[str, list[str]] = {}

    for category, language_terms in terms.items():
        matches = [
            term
            for term in language_terms["english"] + language_terms["tamil"]
            if _normalize_phrase(term) in _normalize_phrase(combined)
        ]
        if matches:
            scores[category] = len(matches)
            evidence[category] = matches

    needs_review = confidence is None or confidence < 50 or not english_translation
    if not scores:
        category = "other_unknown"
        explanation = (
            "The available text does not clearly match a supported sign category. "
            "Check the original sign for its complete message."
        )
        needs_review = True
        matched_evidence: list[str] = []
    else:
        best_score = max(scores.values())
        leaders = [category for category, score in scores.items() if score == best_score]
        if len(leaders) != 1:
            category = "other_unknown"
            explanation = (
                "The recognized text could fit more than one sign category. "
                "Check the original sign before relying on this interpretation."
            )
            needs_review = True
            matched_evidence = [
                term for leader in leaders for term in evidence[leader]
            ]
        else:
            category = leaders[0]
            label = CATEGORY_LABELS[category]
            matched_evidence = evidence[category]
            if english_translation:
                explanation = (
                    f"This appears to be a {label.lower()} sign based on the recognized "
                    "text and translation. Check the original sign for its full message."
                )
            else:
                explanation = (
                    f"The recognized Tamil text contains a cue associated with "
                    f"{label.lower()}. A translation is unavailable, so confirm the "
                    "meaning on the original sign."
                )

    if confidence is not None and confidence < 30:
        explanation = (
            "The recognized text is uncertain. " + explanation
        )
        needs_review = True

    return {
        "category": category,
        "context": explanation,
        "needs_review": needs_review,
        "evidence": matched_evidence,
    }


def lookup_verified_phrase(tamil_text: str) -> dict[str, str] | None:
    """Find an exact verified phrase or a complete OCR line; never infer a fragment."""
    normalized = _normalize_phrase(tamil_text)
    normalized_lines = [
        _normalize_phrase(line)
        for line in tamil_text.splitlines()
        if _normalize_phrase(line)
    ]
    for phrase in _load_json(VERIFIED_PHRASES_PATH):
        variants = [phrase["tamil"], *phrase.get("aliases", [])]
        if any(
            normalized == _normalize_phrase(variant)
            or _normalize_phrase(variant) in normalized_lines
            for variant in variants
        ):
            return phrase
    return None


def verified_phrase_matched_canonical_text(
    tamil_text: str,
    phrase: dict[str, Any],
) -> bool:
    """Tell whether a reviewed match used its canonical spelling or an OCR alias."""
    canonical = _normalize_phrase(phrase["tamil"])
    return _normalize_phrase(tamil_text) == canonical or canonical in [
        _normalize_phrase(line)
        for line in tamil_text.splitlines()
        if _normalize_phrase(line)
    ]


def verified_phrase_matches_full_text(
    tamil_text: str,
    phrase: dict[str, Any],
) -> bool:
    """Check whether the complete OCR text, ignoring punctuation, is the phrase."""
    return _normalize_phrase(tamil_text) == _normalize_phrase(phrase["tamil"])


def _normalize_phrase(text: str) -> str:
    characters = []
    for character in text.casefold():
        category = unicodedata.category(character)
        if category[0] in "LMN":
            characters.append(character)
        elif character.isspace():
            characters.append(" ")
    return re.sub(r"\s+", " ", "".join(characters)).strip()
