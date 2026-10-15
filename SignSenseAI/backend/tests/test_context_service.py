import unittest

from backend.services.context_service import (
    classify_sign,
    lookup_verified_phrase,
    verified_phrase_matched_canonical_text,
)


class ContextServiceTests(unittest.TestCase):
    def test_healthcare_text_is_classified_from_evidence(self):
        result = classify_sign(
            "மருத்துவமனை",
            english_translation="Hospital",
            confidence=90,
        )

        self.assertEqual(result["category"], "healthcare")
        self.assertFalse(result["needs_review"])
        self.assertEqual(
            result["context"],
            "This sign identifies a hospital or medical facility.",
        )
        self.assertIn("Hospital", lookup_verified_phrase("மருத்துவமனை")["english"])

    def test_ambiguous_or_empty_text_requires_review(self):
        ambiguous = classify_sign("மருத்துவமனை", confidence=None)
        empty = classify_sign("")

        self.assertTrue(ambiguous["needs_review"])
        self.assertTrue(empty["needs_review"])
        self.assertEqual(empty["category"], "other_unknown")

    def test_unknown_sign_does_not_invent_details(self):
        result = classify_sign("அ", english_translation="A", confidence=80)

        self.assertEqual(result["category"], "other_unknown")
        self.assertIn("does not clearly match", result["context"])
        self.assertNotIn("distance", result["context"].lower())

    def test_verified_phrase_lookup_ignores_ocr_punctuation_and_line_breaks(self):
        result = lookup_verified_phrase("! குப்பை\nபோடாதே |")
        context = classify_sign(
            "! குப்பை\nபோடாதே |",
            english_translation="Do not litter.",
            confidence=90,
        )

        self.assertEqual(result["english"], "Do not litter.")
        self.assertEqual(result["transliteration"], "Kuppai podaathae")
        self.assertEqual(context["category"], "warnings_prohibitions")
        self.assertEqual(context["context"], "This sign prohibits littering.")

    def test_fertility_center_phrase_and_observed_ocr_alias_are_reviewed(self):
        canonical = lookup_verified_phrase("கருத்தரித்தல் மையம்")
        observed_alias = lookup_verified_phrase("க த்தமித்தல் மையம்")

        self.assertIsNotNone(canonical)
        self.assertEqual(canonical["english"], "Fertility Centre")
        self.assertTrue(
            verified_phrase_matched_canonical_text("கருத்தரித்தல் மையம்", canonical)
        )
        self.assertIsNotNone(observed_alias)
        self.assertEqual(observed_alias["english"], "Fertility Centre")
        self.assertFalse(
            verified_phrase_matched_canonical_text("க த்தமித்தல் மையம்", observed_alias)
        )
        multiline = lookup_verified_phrase("ஐஸ்வர்யா\nக த்தமித்தல் மையம்\nகனவு")
        self.assertEqual(multiline["english"], "Fertility Centre")
        context = classify_sign(
            "கருத்தரித்தல் மையம்",
            english_translation="Fertility Centre",
            confidence=90,
        )
        self.assertEqual(context["category"], "healthcare")
        self.assertIn("fertility centre", context["context"])

    def test_quiet_sign_phrase_uses_reviewed_translation_and_context(self):
        phrase = lookup_verified_phrase("அமைதியாக இருங்கள்")
        result = classify_sign(
            "அமைதியாக இருங்கள்",
            english_translation="Please remain quiet.",
            confidence=88.5,
        )

        self.assertEqual(phrase["english"], "Please remain quiet.")
        self.assertEqual(phrase["transliteration"], "Amaithiyaaga irungal")
        self.assertEqual(result["category"], "warnings_prohibitions")
        self.assertEqual(result["context"], "This sign asks people to remain quiet.")


if __name__ == "__main__":
    unittest.main()
