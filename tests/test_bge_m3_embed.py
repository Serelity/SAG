import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ragflow_style_pipeline.bge_m3_embed import write_embedding_outputs


class TestBgeM3Embed(unittest.TestCase):
    def test_write_embedding_outputs_preserves_multiview_sidecar(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vector_path = Path(tmpdir) / "vectors.npy"
            meta_path = Path(tmpdir) / "vectors.meta.jsonl"
            documents = [
                {
                    "doc_id": "order_a",
                    "case_content_clean": "流动摊贩占道经营",
                    "case_goal_clean": "希望清理",
                    "embedding_text": "诉求内容：流动摊贩占道经营\n诉求目标：希望清理",
                    "display_text": "诉求内容：流动摊贩占道经营\n所属区域：常州市 / 武进区",
                    "text": "诉求内容：流动摊贩占道经营\n所属区域：常州市 / 武进区",
                    "metadata": {"area_code_area": "武进区", "type3": "无照经营游商"},
                }
            ]
            vectors = np.array([[1.0, 0.0]], dtype=np.float32)

            write_embedding_outputs(documents, vectors, vector_path, meta_path)

            saved_vectors = np.load(vector_path)
            saved_meta = json.loads(meta_path.read_text(encoding="utf-8").strip())

            self.assertEqual(saved_vectors.shape, (1, 2))
            self.assertEqual(saved_meta["doc_id"], "order_a")
            self.assertEqual(saved_meta["case_content_clean"], "流动摊贩占道经营")
            self.assertEqual(
                saved_meta["embedding_text"],
                "诉求内容：流动摊贩占道经营\n诉求目标：希望清理",
            )
            self.assertEqual(
                saved_meta["display_text"],
                "诉求内容：流动摊贩占道经营\n所属区域：常州市 / 武进区",
            )
            self.assertEqual(saved_meta["text"], saved_meta["display_text"])
            self.assertEqual(saved_meta["metadata"]["type3"], "无照经营游商")


if __name__ == "__main__":
    unittest.main()
