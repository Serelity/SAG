import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from ragflow_style_pipeline.search_jsonl import main


def _write_jsonl(path, documents):
    path.write_text(
        "\n".join(json.dumps(document, ensure_ascii=False) for document in documents) + "\n",
        encoding="utf-8",
    )


class TestSearchJsonlCli(unittest.TestCase):
    def test_cli_prints_ranked_results_and_writes_json_output(self):
        documents = [
            {
                "doc_id": "salary",
                "text": "诉求内容：工地拖欠工资，工资一直未发。",
                "metadata": {
                    "order_id": "HLW123456789012345",
                    "area_code_area": "武进区",
                    "type1": "民生保障",
                },
            },
            {
                "doc_id": "noise",
                "text": "诉求内容：夜间噪音扰民。",
                "metadata": {"area_code_area": "天宁区", "type1": "环境保护"},
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "docs.jsonl"
            output_path = Path(tmpdir) / "results.json"
            _write_jsonl(input_path, documents)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                main(
                    [
                        "--input",
                        str(input_path),
                        "--query",
                        "拖欠工资",
                        "--top-k",
                        "1",
                        "--output",
                        str(output_path),
                    ]
                )

            self.assertIn("Rank 1", stdout.getvalue())
            self.assertIn("salary", stdout.getvalue())
            saved = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["query"], "拖欠工资")
            self.assertEqual(saved["results"][0]["doc_id"], "salary")
            self.assertNotIn("order_id", saved["results"][0]["metadata"])

    def test_cli_redacts_snippet_before_printing(self):
        documents = [
            {
                "doc_id": "salary",
                "text": "服务对象（姓名：张三，身份证：[身份证号]）反映工地拖欠工资。",
                "metadata": {"area_code_area": "金坛区", "type1": "民生保障"},
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "docs.jsonl"
            output_path = Path(tmpdir) / "results.json"
            _write_jsonl(input_path, documents)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                main(
                    [
                        "--input",
                        str(input_path),
                        "--query",
                        "拖欠工资",
                        "--top-k",
                        "1",
                        "--output",
                        str(output_path),
                    ]
                )

            output = stdout.getvalue()
            self.assertIn("姓名：[姓名]", output)
            self.assertNotIn("张三", output)

            saved = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertIn("姓名：[姓名]", saved["results"][0]["snippet"])
            self.assertNotIn("张三", json.dumps(saved, ensure_ascii=False))
            self.assertNotIn("text", saved["results"][0])


if __name__ == "__main__":
    unittest.main()
