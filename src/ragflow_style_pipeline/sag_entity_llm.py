"""LLM-assisted SAG entity extraction for 12345 work orders."""

import argparse
import json
import os
import re
import time
from pathlib import Path

from ragflow_style_pipeline.sag_db import read_source_rows
from ragflow_style_pipeline.sag_entities import SagEntityLink, clean_value, deduplicate_entity_links
from ragflow_style_pipeline.sag_entity_schema import normalize_llm_entity_value, validate_llm_candidate


ENTITY_INSTRUCTIONS = """你是 12345 工单的 SAG 检索实体抽取器。
任务只服务后续 SQL join 检索，不做完整治理分析。
只允许输出这些 entity_type：
- problem_object：问题对象，例如流动摊贩、商贩、车辆、垃圾、井盖
- problem_behavior：问题行为，例如占道经营、影响通行、扰民、乱停放、损坏
- area：区县/开发区
- street：街道/镇
- road：道路/街/大道/巷/桥/线的具体名称
- intersection：两个道路形成的路口/交叉口/交界处
- poi：小区、菜场、学校、医院、商场、公园等具体地点
不要输出满意度、处理结果、风险等级、部门职责、完整摘要。
实体必须来自原文，evidence_span 必须能在对应 source_field 的原文中找到。
不要把“道路”“路”“街”“小区”“关于小区”“关于道路”“常州12345热线”这类泛词作为实体。
只输出 JSON，不输出解释文字。
JSON 格式：
{"entities":[{"entity_type":"road","entity_value":"广成路","source_field":"case_content_clean","evidence_span":"广成路","confidence":0.9}]}
"""


def _truncate(value, max_chars):
    value = clean_value(value)
    if len(value) <= max_chars:
        return value
    return value[:max_chars]


def build_extraction_prompt(order, config):
    """Build a deterministic extraction prompt for one normalized source order."""
    max_text_chars = int(config.get("max_text_chars", 2200))
    fields = {
        "title_clean": _truncate(order.get("title_clean"), 300),
        "case_content_clean": _truncate(order.get("case_content_clean"), max_text_chars),
        "case_goal_clean": _truncate(order.get("case_goal_clean"), 500),
        "address_detail_clean": _truncate(order.get("address_detail_clean"), 500),
    }
    payload = json.dumps(fields, ensure_ascii=False, indent=2)
    return f"{ENTITY_INSTRUCTIONS}\n输入工单字段：\n{payload}\n只输出 JSON："


def parse_llm_json(text):
    """Parse the first JSON object from model output."""
    text = clean_value(text)
    text = re.sub(r"^```(?:json)?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return {"entities": []}
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {"entities": []}
    if not isinstance(parsed, dict):
        return {"entities": []}
    entities = parsed.get("entities")
    if not isinstance(entities, list):
        parsed["entities"] = []
    return parsed


def candidate_to_link(doc_id, candidate):
    """Convert one validated LLM candidate into a SagEntityLink."""
    entity_type = clean_value(candidate.get("entity_type"))
    entity_value = normalize_llm_entity_value(entity_type, candidate.get("entity_value"))
    source_field = clean_value(candidate.get("source_field")) or "case_content_clean"
    evidence_span = clean_value(candidate.get("evidence_span") or candidate.get("matched_text") or entity_value)
    confidence = float(candidate.get("confidence") or 0.0)
    return SagEntityLink(
        doc_id=doc_id,
        entity_type=entity_type,
        entity_value=entity_value,
        normalized_value=entity_value,
        source_field=source_field,
        source_channel="llm",
        confidence=confidence,
        matched_text=evidence_span,
    )


def extract_links_from_llm_response(order, response_text, config):
    """Validate model response and return SAG entity links plus rejected candidates."""
    parsed = parse_llm_json(response_text)
    links = []
    rejects = []
    doc_id = clean_value(order.get("doc_id"))
    for candidate in parsed.get("entities", []):
        if not isinstance(candidate, dict):
            rejects.append({"candidate": candidate, "reason": "candidate_not_object", "doc_id": doc_id})
            continue
        ok, reason = validate_llm_candidate(candidate, order, config)
        if ok:
            links.append(candidate_to_link(doc_id, candidate))
        else:
            rejects.append({"candidate": candidate, "reason": reason, "doc_id": doc_id})
    return deduplicate_entity_links(links), rejects


def _select_torch_dtype(torch_module):
    dtype_name = os.environ.get("ENTITY_LLM_DTYPE", "").strip().lower()
    if dtype_name == "float32":
        return torch_module.float32
    if dtype_name == "bfloat16":
        return torch_module.bfloat16
    if dtype_name == "float16":
        return torch_module.float16
    if torch_module.cuda.is_available():
        return torch_module.float16
    return torch_module.float32


def _config_with_env_overrides(config):
    """Return a config copy with server-side environment overrides applied."""
    merged = dict(config)
    int_overrides = {
        "BATCH_SIZE": "batch_size",
        "PROGRESS_EVERY": "progress_every",
        "MAX_NEW_TOKENS": "max_new_tokens",
        "MAX_TEXT_CHARS": "max_text_chars",
    }
    for env_name, config_name in int_overrides.items():
        value = clean_value(os.environ.get(env_name))
        if value:
            merged[config_name] = int(value)
    return merged


def _to_input_text(tokenizer, prompt, enable_thinking=False):
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=bool(enable_thinking),
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def load_local_generator(model_path, enable_thinking=False):
    """Load a local causal LM for deterministic batched JSON extraction."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Local model path not found: {model_path}. "
            "Run MODEL_ID=Qwen/Qwen3-4B MODEL_DIR=models/Qwen3-4B bash scripts/download_entity_model.sh first."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=_select_torch_dtype(torch),
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )

    def generate(prompts, max_new_tokens=256, temperature=0.0):
        single_input = isinstance(prompts, str)
        prompt_list = [prompts] if single_input else list(prompts)
        if not prompt_list:
            return "" if single_input else []

        input_texts = [_to_input_text(tokenizer, prompt, enable_thinking=enable_thinking) for prompt in prompt_list]
        inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        do_sample = float(temperature) > 0.0
        generate_kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = float(temperature)
        with torch.inference_mode():
            outputs = model.generate(**inputs, **generate_kwargs)
        generated = outputs[:, inputs.input_ids.shape[-1] :]
        responses = tokenizer.batch_decode(generated, skip_special_tokens=True)
        return responses[0] if single_input else responses

    return generate


def link_to_json(link):
    """Serialize a SagEntityLink to JSONL."""
    return {
        "doc_id": link.doc_id,
        "entity_type": link.entity_type,
        "entity_value": link.entity_value,
        "normalized_value": link.normalized_value,
        "source_field": link.source_field,
        "source_channel": link.source_channel,
        "confidence": link.confidence,
        "matched_text": link.matched_text,
    }


def run_extraction(input_path, output_path, rejects_path, model_path, config, limit=None):
    """Run LLM entity extraction over source rows and write entity-link JSONL."""
    config = _config_with_env_overrides(config)
    generator = load_local_generator(model_path, enable_thinking=bool(config.get("enable_thinking", False)))
    rows = read_source_rows(input_path, limit=limit)
    batch_size = max(1, int(config.get("batch_size", 8)))
    progress_every = max(1, int(config.get("progress_every", batch_size)))
    max_new_tokens = int(config.get("max_new_tokens", 256))
    temperature = float(config.get("temperature", 0.0))
    written_links = 0
    written_rejects = 0
    processed = 0
    next_progress = progress_every
    started_at = time.time()
    output_path = Path(output_path)
    rejects_path = Path(rejects_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejects_path.parent.mkdir(parents=True, exist_ok=True)

    def print_progress(done=False):
        elapsed = time.time() - started_at
        rate = processed / elapsed if elapsed > 0 else 0.0
        remaining = len(rows) - processed
        eta = remaining / rate if rate > 0 else None
        print(
            json.dumps(
                {
                    "processed": processed,
                    "total": len(rows),
                    "batch_size": batch_size,
                    "links": written_links,
                    "rejects": written_rejects,
                    "elapsed_seconds": round(elapsed, 1),
                    "orders_per_second": round(rate, 3),
                    "eta_seconds": round(eta, 1) if eta is not None else None,
                    "done": done,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    with output_path.open("w", encoding="utf-8") as output_file, rejects_path.open("w", encoding="utf-8") as rejects_file:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            prompts = [build_extraction_prompt(order, config) for order in batch]
            responses = generator(prompts, max_new_tokens=max_new_tokens, temperature=temperature)
            if isinstance(responses, str):
                responses = [responses]
            if len(responses) != len(batch):
                raise RuntimeError(f"Generator returned {len(responses)} responses for {len(batch)} prompts")

            for order, response in zip(batch, responses):
                links, rejects = extract_links_from_llm_response(order, response, config)
                for link in links:
                    output_file.write(json.dumps(link_to_json(link), ensure_ascii=False) + "\n")
                    written_links += 1
                for reject in rejects:
                    rejects_file.write(json.dumps(reject, ensure_ascii=False) + "\n")
                    written_rejects += 1
                processed += 1

            if processed >= next_progress or processed == len(rows):
                output_file.flush()
                rejects_file.flush()
                print_progress(done=processed == len(rows))
                while next_progress <= processed:
                    next_progress += progress_every

    elapsed = time.time() - started_at
    return {
        "orders_processed": processed,
        "links_written": written_links,
        "rejects_written": written_rejects,
        "batch_size": batch_size,
        "elapsed_seconds": round(elapsed, 1),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run local LLM SAG entity extraction.")
    parser.add_argument("--input", required=True, help="Input TSV or JSONL source rows.")
    parser.add_argument("--output", required=True, help="Output SagEntityLink JSONL.")
    parser.add_argument("--rejects", required=True, help="Rejected candidate JSONL.")
    parser.add_argument("--config", required=True, help="Extraction config JSON.")
    parser.add_argument("--model-path", required=True, help="Local model directory.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum source rows to process.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    summary = run_extraction(args.input, args.output, args.rejects, args.model_path, config, limit=args.limit)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
