import unittest

from ragflow_style_pipeline.pii_redactor import redact_text


class TestPiiRedactor(unittest.TestCase):
    def test_redacts_mainland_phone_number(self):
        phone = "138" + "0013" + "8000"

        text, counts = redact_text("联系电话" + phone + "，请处理")

        self.assertEqual(text, "联系电话[手机号]，请处理")
        self.assertEqual(counts["phone"], 1)
        self.assertEqual(counts["id_card"], 0)

    def test_redacts_phone_number_with_leading_zero(self):
        phone = "0" + "134" + "0197" + "1321"

        text, counts = redact_text("电话号码：" + phone)

        self.assertEqual(text, "电话号码：[手机号]")
        self.assertEqual(counts["phone"], 1)

    def test_redacts_phone_number_with_extra_trailing_digit(self):
        phone = "180" + "0611" + "0955" + "5"

        text, counts = redact_text("负责人电话：" + phone)

        self.assertEqual(text, "负责人电话：[手机号]")
        self.assertEqual(counts["phone"], 1)

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

    def test_redacts_medium_numeric_identifier(self):
        number = "621691" + "050338" + "6481"

        text, counts = redact_text("银行卡号：" + number)

        self.assertEqual(text, "银行卡号：[数字编号]")
        self.assertEqual(counts["numeric_id"], 1)

    def test_redacts_alpha_numeric_identifier(self):
        number = "TB" + "213440" + "753842"

        text, counts = redact_text("商家编号：" + number)

        self.assertEqual(text, "商家编号：[业务编号]")
        self.assertEqual(counts["alnum_id"], 1)

    def test_redacts_alpha_numeric_identifier_next_to_chinese_text(self):
        number = "JDX" + "036920" + "695970"

        text, counts = redact_text("快递" + number + "一直不更新")

        self.assertEqual(text, "快递[业务编号]一直不更新")
        self.assertEqual(counts["alnum_id"], 1)

    def test_redacts_name_after_explicit_name_label(self):
        text, counts = redact_text("服务对象（姓名：李长泉，身份证：[身份证号]）反映拖欠工资")

        self.assertEqual(text, "服务对象（姓名：[姓名]，身份证：[身份证号]）反映拖欠工资")
        self.assertEqual(counts["name"], 1)

    def test_none_becomes_empty_text(self):
        text, counts = redact_text(None)

        self.assertEqual(text, "")
        self.assertEqual(counts["phone"], 0)
        self.assertEqual(counts["id_card"], 0)


if __name__ == "__main__":
    unittest.main()
