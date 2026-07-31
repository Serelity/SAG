import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ragflow_style_pipeline.vector_search import load_vector_index, vector_search


class TestVectorSearch(unittest.TestCase):
    def test_vector_search_returns_nearest_multiview_document(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vector_path = Path(tmpdir) / "vectors.npy"
            meta_path = Path(tmpdir) / "vectors.meta.jsonl"
            np.save(vector_path, np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
            meta_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "doc_id": "stall",
                                "display_text": "诉求内容：流动摊贩占道经营",
                                "embedding_text": "诉求内容：流动摊贩占道经营",
                                "case_content_clean": "流动摊贩占道经营",
                                "metadata": {"type3": "无照经营游商", "area_code_area": "武进区"},
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "doc_id": "noise",
                                "display_text": "诉求内容：噪音扰民",
                                "embedding_text": "诉求内容：噪音扰民",
                                "case_content_clean": "噪音扰民",
                                "metadata": {"type3": "社会生活噪声", "area_code_area": "天宁区"},
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            index = load_vector_index(vector_path, meta_path)
            results = vector_search(index, np.array([0.9, 0.1], dtype=np.float32), top_k=1)

            self.assertEqual(results[0]["doc_id"], "stall")
            self.assertEqual(results[0]["retriever"], "vector")
            self.assertEqual(results[0]["text"], "诉求内容：流动摊贩占道经营")
            self.assertEqual(results[0]["case_content_clean"], "流动摊贩占道经营")
            self.assertIn("vector_score", results[0])


if __name__ == "__main__":
    unittest.main()
