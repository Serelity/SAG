"""Synthetic tests for fail-closed configuration and V100 environment switches."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ragflow_style_pipeline.pipeline import PipelineError, load_config, model_manifest
from ragflow_style_pipeline.vllm_backend import BackendError, validate_environment


SAFE_ENV = {
    "VLLM_USE_V1": "0",
    "VLLM_ATTENTION_BACKEND": "XFORMERS",
    "VLLM_ENABLE_PREFIX_CACHING": "0",
    "VLLM_ENABLE_CHUNKED_PREFILL": "0",
    "VLLM_ENFORCE_EAGER": "0",
    "VLLM_LOGGING_LEVEL": "WARNING",
}


class ConfigBackendTests(unittest.TestCase):
    def _config_value(self):
        source = Path(__file__).parents[1] / "configs" / "entity_extraction_v1.json"
        return json.loads(source.read_text(encoding="utf-8"))

    def test_config_is_single_strict_contract(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "config.json"
            path.write_text(json.dumps(self._config_value()), encoding="utf-8")
            config = load_config(path)
            self.assertEqual("entity_extraction_v1", config["pipeline_version"])
            for key, bad in (
                ("enable_thinking", True),
                ("enable_prefix_caching", True),
                ("enable_chunked_prefill", True),
                ("enforce_eager", True),
                ("tensor_parallel_size", 2),
            ):
                changed = {**config, key: bad}
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.subTest(key=key), self.assertRaises(PipelineError):
                    load_config(path)

    def test_environment_must_match_v100_contract_exactly(self):
        with patch.dict(os.environ, SAFE_ENV, clear=True):
            validate_environment()
        for key in SAFE_ENV:
            changed = {**SAFE_ENV, key: "1"}
            with self.subTest(key=key), patch.dict(os.environ, changed, clear=True):
                with self.assertRaisesRegex(BackendError, "unsafe_environment"):
                    validate_environment()

    def test_model_manifest_hashes_small_contract_files_not_weight_bytes(self):
        with tempfile.TemporaryDirectory() as root:
            model = Path(root)
            (model / "config.json").write_text('{"model_type":"qwen3"}', encoding="utf-8")
            (model / "tokenizer.json").write_text('{"synthetic":true}', encoding="utf-8")
            weight = model / "model.safetensors"
            weight.write_bytes(b"AAAA")
            first = model_manifest(model)
            weight.write_bytes(b"BBBB")
            second = model_manifest(model)
            self.assertEqual(first["fingerprint"], second["fingerprint"])
            (model / "config.json").write_text(
                '{"model_type":"qwen3","synthetic_revision":2}', encoding="utf-8"
            )
            third = model_manifest(model)
            self.assertNotEqual(second["fingerprint"], third["fingerprint"])
            self.assertEqual(
                "config_tokenizer_code_content_plus_weight_name_size_v1",
                first["method"],
            )


if __name__ == "__main__":
    unittest.main()
