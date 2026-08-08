"""The only production generator: fail-closed vLLM settings proven on V100."""

from __future__ import annotations

import os
import time


_REQUIRED_ENVIRONMENT = {
    "VLLM_USE_V1": "0",
    "VLLM_ATTENTION_BACKEND": "XFORMERS",
    "VLLM_ENABLE_PREFIX_CACHING": "0",
    "VLLM_ENABLE_CHUNKED_PREFILL": "0",
    "VLLM_ENFORCE_EAGER": "0",
    "VLLM_LOGGING_LEVEL": "WARNING",
}


class BackendError(RuntimeError):
    """A safe backend configuration or generation failure code."""


def validate_environment() -> None:
    for name, expected in _REQUIRED_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise BackendError("unsafe_environment:" + name)


class VLLMBackend:
    def __init__(self, model_path: str, config: dict):
        validate_environment()
        if (
            config["dtype"] != "float16"
            or config["tensor_parallel_size"] != 1
            or config["enable_thinking"] is not False
            or config["enable_prefix_caching"] is not False
            or config["enable_chunked_prefill"] is not False
            or config["enforce_eager"] is not False
        ):
            raise BackendError("unsafe_backend_config")

        from vllm import LLM

        self.config = config
        self.llm = LLM(
            model=model_path,
            tokenizer=model_path,
            trust_remote_code=True,
            dtype="float16",
            tensor_parallel_size=1,
            gpu_memory_utilization=config["gpu_memory_utilization"],
            max_model_len=config["max_model_len"],
            max_num_seqs=config["max_num_seqs"],
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            enforce_eager=False,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def generate(self, prompts: list[str], max_tokens: int) -> list[dict]:
        from vllm import SamplingParams

        rendered_prompts = []
        for prompt in prompts:
            try:
                rendered = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError as exc:
                # Qwen3 deployment must understand the thinking switch; silently omitting it is unsafe.
                raise BackendError("tokenizer_missing_thinking_switch") from exc
            rendered_prompts.append(rendered)

        sampling = SamplingParams(
            temperature=self.config["temperature"],
            seed=self.config["seed"],
            max_tokens=max_tokens,
            skip_special_tokens=True,
        )
        started = time.perf_counter()
        outputs = self.llm.generate(rendered_prompts, sampling, use_tqdm=False)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        if len(outputs) != len(prompts):
            raise BackendError("generation_count_mismatch")

        memory = _memory_stats()
        latency_share_ms = round(elapsed_ms / max(1, len(outputs)), 3)
        rows = []
        for output in outputs:
            if not output.outputs:
                raise BackendError("generation_choice_missing")
            choice = output.outputs[0]
            rows.append(
                {
                    "text": choice.text,
                    "finish_reason": str(choice.finish_reason or "unknown"),
                    "input_tokens": len(output.prompt_token_ids or []),
                    "output_tokens": len(choice.token_ids or []),
                    "latency_share_ms": latency_share_ms,
                    **memory,
                }
            )
        return rows


def _memory_stats() -> dict[str, float]:
    try:
        import torch

        scale = 1024**3
        return {
            "gpu_peak_allocated_gb": round(torch.cuda.max_memory_allocated() / scale, 3),
            "gpu_peak_reserved_gb": round(torch.cuda.max_memory_reserved() / scale, 3),
        }
    except Exception:
        return {"gpu_peak_allocated_gb": -1.0, "gpu_peak_reserved_gb": -1.0}
