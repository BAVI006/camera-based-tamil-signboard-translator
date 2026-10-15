import unittest

from indic_transliteration import sanscript

from backend.services.transliteration_service import transliterate_tamil


class TransliterationTests(unittest.TestCase):
    def test_transliterates_the_project_example_using_verified_library_api(self):
        tamil = "மருத்துவமனை"
        expected = sanscript.transliterate(
            tamil, sanscript.TAMIL, sanscript.ITRANS
        ).replace("ன", "n")
        actual = transliterate_tamil(tamil)

        self.assertEqual(actual, expected)
        self.assertTrue(actual)
        self.assertTrue(actual.isascii())

    def test_empty_or_non_tamil_text_has_no_tamil_transliteration(self):
        self.assertIsNone(transliterate_tamil(""))
        self.assertIsNone(transliterate_tamil("   "))
        self.assertIsNone(transliterate_tamil("Hospital"))
