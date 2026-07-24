import unittest

from ragflow_style_pipeline.pii_redactor import redact_text


class TestPiiRedactor(unittest.TestCase):
    def test_redacts_mainland_phone_number(self):
        phone = "138" + "0013" + "8000"

        text, counts = redact_text("联系电话" + phone + "，请处理")

        self.assertEqual(text, "联系电话[手机号]，请处理")
        self.assertEqual(counts["phone"], 1)
        self.assertEqual(counts["id_card"], 0)

    def test_redacts_18_digit_id_number(self):
        id_card = "320400" + "19900101" + "123" + "X"

        text, counts = redact_text("身份证号" + id_card + "可以办理吗")

        self.assertEqual(text, "身份证号[身份证号]可以办理吗")
        self.assertEqual(counts["phone"], 0)
        self.assertEqual(counts["id_card"], 1)

    def test_redacts_very_long_numeric_identifier(self):
        long_number = "320412" + "20160908" + "008312"

        text, counts = redact_text("残疾证号：" + long_number)

        self.assertEqual(text, "残疾证号：[长数字编号]")
        self.assertEqual(counts["long_number"], 1)

    def test_none_becomes_empty_text(self):
        text, counts = redact_text(None)

        self.assertEqual(text, "")
        self.assertEqual(counts["phone"], 0)
        self.assertEqual(counts["id_card"], 0)


if __name__ == "__main__":
    unittest.main()
