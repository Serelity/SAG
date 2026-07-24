import unittest

from ragflow_style_pipeline.text_tokenizer import tokenize


class TestTextTokenizer(unittest.TestCase):
    def test_tokenizes_chinese_bigrams_and_ascii_terms(self):
        tokens = tokenize("武进区 夜间摆摊扰民 GPT-5 2024")

        self.assertIn("武进", tokens)
        self.assertIn("进区", tokens)
        self.assertIn("夜间", tokens)
        self.assertIn("摆摊", tokens)
        self.assertIn("扰民", tokens)
        self.assertIn("gpt", tokens)
        self.assertIn("5", tokens)
        self.assertIn("2024", tokens)

    def test_ignores_blank_text(self):
        self.assertEqual(tokenize("  \n\t  "), [])

    def test_keeps_single_chinese_character_segment(self):
        self.assertEqual(tokenize("水"), ["水"])


if __name__ == "__main__":
    unittest.main()
