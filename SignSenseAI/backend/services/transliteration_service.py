"""Tamil-to-Latin transliteration, deliberately separate from translation."""

from indic_transliteration import sanscript


_UNMAPPED_TAMIL_LETTERS = {
    "ன": "n",
    "ஜ": "j",
    "ஷ": "sh",
    "ஸ": "s",
    "ஹ": "h",
}


def transliterate_tamil(text: str) -> str | None:
    """Return scholarly ITRANS transliteration, or None when no Tamil is present."""
    if not text or not text.strip():
        return None

    tamil_present = any("\u0b80" <= character <= "\u0bff" for character in text)
    if not tamil_present:
        return None

    result = sanscript.transliterate(text.strip(), sanscript.TAMIL, sanscript.ITRANS)
    for tamil_letter, latin_letters in _UNMAPPED_TAMIL_LETTERS.items():
        result = result.replace(tamil_letter, latin_letters)
    return result
