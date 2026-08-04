import json
import tempfile
import unittest
from pathlib import Path

from ragflow_style_pipeline.sag_semantic_projection import project_semantic_record, project_semantics_file


class TestSemanticProjection(unittest.TestCase):
    def test_projects_entities_and_discourse_without_model_confidence(self):
        order = {"doc_id":"order_1","metadata":{"service_object_type":"求助","area_code_area":"钟楼区","area_code_street":"永红街道"}}
        record = {"doc_id":"order_1","event":{"summary":"市民反映和平路路灯连续三天不亮，希望维修"},"entities":{
            "problem_objects":[{"surface":"路灯","canonical":"路灯","source_field":"case_content_clean","evidence":"路灯"}],
            "problem_behaviors":[{"surface":"连续三天不亮","canonical":"照明故障","source_field":"case_content_clean","evidence":"连续三天不亮"}],
            "roads":[{"surface":"和平路","canonical":"和平路","source_field":"case_content_clean","evidence":"和平路"}],"intersections":[],"pois":[]},
            "discourse":{"intents":[{"label":"求助","evidence":"希望维修"}],"emotions":[],"satisfaction":{"label":"unknown","target":"","evidence":""},"urgency":{"level":"normal","evidence":""}},
            "validation":{"status":"accepted","warnings":[]},"model_run":{"prompt_version":"sag_semantic_v2"}}
        event, links, discourse = project_semantic_record(record, order)
        self.assertEqual(event["event_text"], record["event"]["summary"])
        self.assertEqual({(row["entity_type"],row["normalized_value"]) for row in links},{("problem_object","路灯"),("problem_behavior","照明故障"),("road","和平路")})
        self.assertTrue(all("confidence" not in row for row in links))
        self.assertEqual(discourse["declared_intent"], "求助")
        self.assertEqual(discourse["satisfaction"], "unknown")

    def test_projection_file_writes_atomic_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp=Path(tmpdir); orders=tmp/"orders.jsonl"; semantics=tmp/"semantics.jsonl"
            orders.write_text(json.dumps({"doc_id":"d1","case_content_clean":"和平路路灯不亮"},ensure_ascii=False)+"\n",encoding="utf-8")
            semantics.write_text(json.dumps({"doc_id":"d1","event":{"summary":"和平路路灯不亮"},"entities":{"problem_objects":[],"problem_behaviors":[],"roads":[],"intersections":[],"pois":[]},"discourse":{},"validation":{"status":"accepted"},"model_run":{}},ensure_ascii=False)+"\n",encoding="utf-8")
            report=project_semantics_file(semantics,orders,tmp/"links.jsonl",tmp/"disc.jsonl")
            self.assertEqual(report["records"],1)
            self.assertTrue((tmp/"links.jsonl").exists())
            self.assertEqual(len((tmp/"disc.jsonl").read_text(encoding="utf-8").splitlines()),1)


if __name__ == "__main__": unittest.main()
