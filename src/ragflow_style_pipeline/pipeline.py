"""Durable batched primary/repair extraction for the single v1 contract."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Iterator

from .constants import (
    CHECKPOINT_PRIVATE_NAME,
    CONTRACT_PRIVATE_NAME,
    DIAGNOSTICS_SAFE_NAME,
    DOCUMENT_PRIVATE_NAME,
    DOCUMENT_SCHEMA_VERSION,
    ENTITIES_PRIVATE_NAME,
    ENTITY_SCHEMA_VERSION,
    LOCK_PRIVATE_NAME,
    METADATA_SOURCES,
    PII_REDACTION_VERSION,
    PIPELINE_VERSION,
    PREPARE_SAFE_NAME,
    REJECTS_PRIVATE_NAME,
    REJECT_SCHEMA_VERSION,
)
from .entity_prompt import primary_prompt, prompt_fingerprint, repair_prompt
from .entity_schema import EntitySchemaError, parse_model_output
from .grounding import ground_payload
from .pii_redactor import residual_pii_codes
from .work_order import (
    atomic_write_json,
    canonical_json_bytes,
    clean_content_hash,
    file_sha256,
    build_rag_text,
)


_CONFIG_KEYS = frozenset(
    {
        "pipeline_version",
        "model_name",
        "batch_size",
        "primary_max_tokens",
        "repair_max_tokens",
        "max_input_chars",
        "temperature",
        "seed",
        "dtype",
        "tensor_parallel_size",
        "max_model_len",
        "max_num_seqs",
        "gpu_memory_utilization",
        "enable_thinking",
        "enable_prefix_caching",
        "enable_chunked_prefill",
        "enforce_eager",
    }
)
_TELEMETRY_KEYS = (
    "finish_reason",
    "input_tokens",
    "output_tokens",
    "latency_share_ms",
    "gpu_peak_allocated_gb",
    "gpu_peak_reserved_gb",
)
_SAFE_CODE_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")
_DIAGNOSTIC_KEYS = frozenset({"attempt", "outcome", *_TELEMETRY_KEYS})
_MODEL_CONTENT_NAMES = frozenset(
    {
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "tokenizer.model",
    }
)
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt")
_MAX_MANIFEST_CONTENT_BYTES = 64 * 1024 * 1024


class PipelineError(ValueError):
    """A pipeline failure whose message is a non-sensitive error code."""


def canonical_hash(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def load_config(path: Path) -> dict:
    try:
        config = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PipelineError("invalid_config") from exc
    if not isinstance(config, dict) or set(config) != _CONFIG_KEYS:
        raise PipelineError("invalid_config_keys")
    integer_keys = (
        "batch_size",
        "primary_max_tokens",
        "repair_max_tokens",
        "max_input_chars",
        "seed",
        "tensor_parallel_size",
        "max_model_len",
        "max_num_seqs",
    )
    if any(type(config[key]) is not int or config[key] < 0 for key in integer_keys):
        raise PipelineError("invalid_config_integer")
    if any(config[key] <= 0 for key in integer_keys if key != "seed"):
        raise PipelineError("invalid_config_integer")
    if config["pipeline_version"] != PIPELINE_VERSION or config["model_name"] != "Qwen3-4B":
        raise PipelineError("invalid_config_version")
    if (
        type(config["temperature"]) not in (int, float)
        or config["temperature"] != 0
        or config["dtype"] != "float16"
        or config["tensor_parallel_size"] != 1
        or config["batch_size"] > config["max_num_seqs"]
        or config["max_model_len"] < 4096
    ):
        raise PipelineError("unsafe_inference_config")
    for key in (
        "enable_thinking",
        "enable_prefix_caching",
        "enable_chunked_prefill",
        "enforce_eager",
    ):
        if config[key] is not False:
            raise PipelineError("unsafe_inference_config")
    utilization = config["gpu_memory_utilization"]
    if type(utilization) not in (int, float) or not 0.1 <= utilization <= 0.98:
        raise PipelineError("invalid_gpu_memory_utilization")
    return config


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def model_manifest(model_path: Path) -> dict:
    """Bind small config/tokenizer content and weight names/sizes, not weight bytes."""
    root = Path(model_path)
    if not root.is_dir():
        raise PipelineError("missing_model_directory")
    content_files = []
    weight_files = []
    try:
        files = sorted(item for item in root.rglob("*") if item.is_file())
        for item in files:
            relative = item.relative_to(root).as_posix()
            size = item.stat().st_size
            if (
                item.name in _MODEL_CONTENT_NAMES
                or item.name.endswith((".index.json", ".py", ".jinja"))
            ):
                if size > _MAX_MANIFEST_CONTENT_BYTES:
                    raise PipelineError("model_manifest_file_too_large")
                content_files.append(
                    {"name": relative, "size": size, "sha256": _sha256_file(item)}
                )
            if item.name.endswith(_WEIGHT_SUFFIXES):
                weight_files.append({"name": relative, "size": size})
    except OSError as exc:
        raise PipelineError("model_manifest_unreadable") from exc
    if not any(item["name"] == "config.json" for item in content_files):
        raise PipelineError("model_config_missing")
    try:
        root_config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PipelineError("model_config_invalid") from exc
    if not isinstance(root_config, dict) or root_config.get("model_type") != "qwen3":
        raise PipelineError("model_type_not_qwen3")
    if not weight_files:
        raise PipelineError("model_weights_missing")
    manifest = {
        "method": "config_tokenizer_code_content_plus_weight_name_size_v1",
        "content_files": content_files,
        "weight_files": weight_files,
    }
    return {"fingerprint": canonical_hash(manifest), **manifest}


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict]]:
    try:
        source = Path(path).open("r", encoding="utf-8")
    except OSError as exc:
        raise PipelineError("jsonl_unavailable") from exc
    with source:
        try:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    raise PipelineError("blank_jsonl_line")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise PipelineError("jsonl_object_required")
                yield line_number, value
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise PipelineError("invalid_jsonl") from exc


def _validate_document(document: dict) -> None:
    required = {
        "schema_version",
        "pipeline_version",
        "redaction_version",
        "doc_id",
        "content_hash",
        "title_clean",
        "case_content_clean",
        "case_goal_clean",
        "address_detail_clean",
        "rag_text",
        "metadata",
    }
    if set(document) != required:
        raise PipelineError("invalid_document_keys")
    clean_fields = {
        name: document[name]
        for name in ("title_clean", "case_content_clean", "case_goal_clean", "address_detail_clean")
    }
    if (
        document["schema_version"] != DOCUMENT_SCHEMA_VERSION
        or document["pipeline_version"] != PIPELINE_VERSION
        or document["redaction_version"] != PII_REDACTION_VERSION
        or not isinstance(document["doc_id"], str)
        or not document["doc_id"].startswith("order_")
        or any(not isinstance(value, str) for value in clean_fields.values())
        or document["content_hash"] != clean_content_hash(clean_fields)
        or not isinstance(document["rag_text"], str)
        or not isinstance(document["metadata"], dict)
        or set(document["metadata"]) != {*METADATA_SOURCES, "call_month"}
        or any(not isinstance(value, str) for value in document["metadata"].values())
        or document["metadata"]["call_month"]
        != (document["metadata"]["call_time"][:7] if document["metadata"]["call_time"] else "")
        or document["rag_text"] != build_rag_text(clean_fields, document["metadata"])
        or any(residual_pii_codes(value) for value in clean_fields.values())
        or residual_pii_codes(document["rag_text"])
        or any(residual_pii_codes(value) for value in document["metadata"].values())
    ):
        raise PipelineError("invalid_document_contract")


def documents_manifest(documents_path: Path) -> dict:
    digest = hashlib.sha256()
    seen = set()
    count = 0
    for _line_number, document in iter_jsonl(documents_path):
        _validate_document(document)
        doc_id = document["doc_id"]
        if doc_id in seen:
            raise PipelineError("duplicate_document_id")
        seen.add(doc_id)
        identity = [doc_id, document["content_hash"]]
        digest.update(canonical_json_bytes(identity) + b"\n")
        count += 1
    if count == 0:
        raise PipelineError("empty_documents")
    snapshot_sha256, snapshot_bytes = file_sha256(documents_path)
    return {
        "count": count,
        "fingerprint": "sha256:" + digest.hexdigest(),
        "snapshot_sha256": snapshot_sha256,
        "snapshot_bytes": snapshot_bytes,
    }


def _validate_prepare_publication(run_dir: Path, documents: dict) -> None:
    try:
        report = json.loads((run_dir / PREPARE_SAFE_NAME).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PipelineError("prepare_report_invalid") from exc
    if (
        not isinstance(report, dict)
        or report.get("documents_written") != documents["count"]
        or report.get("output_sha256") != documents["snapshot_sha256"]
        or report.get("output_bytes") != documents["snapshot_bytes"]
    ):
        raise PipelineError("prepare_publication_mismatch")


def build_run_contract(
    documents_path: Path, config: dict, model_path: Path, *, run_dir: Path | None = None
) -> dict:
    model = model_manifest(model_path)
    documents = documents_manifest(documents_path)
    if run_dir is not None:
        _validate_prepare_publication(run_dir, documents)
    body = {
        "pipeline_version": PIPELINE_VERSION,
        "entity_schema_version": ENTITY_SCHEMA_VERSION,
        "documents": documents,
        "model_fingerprint": model["fingerprint"],
        "model_fingerprint_method": model["method"],
        "prompt_fingerprint": prompt_fingerprint(),
        "config_fingerprint": canonical_hash(config),
    }
    return {**body, "contract_hash": canonical_hash(body)}


def _sync_jsonl(handle, value: dict) -> None:
    handle.write(canonical_json_bytes(value).decode("utf-8") + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _recover_trailing_partial(path: Path) -> None:
    """Drop only an unterminated final JSONL fragment left by abrupt process death."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("r+b") as stream:
        stream.seek(-1, os.SEEK_END)
        if stream.read(1) == b"\n":
            return
        position = stream.tell() - 1
        while position > 0:
            position -= 1
            stream.seek(position)
            if stream.read(1) == b"\n":
                position += 1
                break
        else:
            position = 0
        stream.truncate(position)
        stream.flush()
        os.fsync(stream.fileno())


def _safe_telemetry(output: dict) -> dict:
    telemetry = {}
    for key in _TELEMETRY_KEYS:
        value = output.get(key)
        if isinstance(value, str) and key == "finish_reason":
            telemetry[key] = value if _SAFE_CODE_RE.fullmatch(value) else "unknown"
        elif type(value) in (int, float) and math.isfinite(value):
            telemetry[key] = value
    return telemetry


def _validate_diagnostic(value: object) -> dict:
    if (
        not isinstance(value, dict)
        or not {"attempt", "outcome"} <= set(value) <= _DIAGNOSTIC_KEYS
        or not isinstance(value.get("attempt"), str)
        or not isinstance(value.get("outcome"), str)
        or not _SAFE_CODE_RE.fullmatch(value["attempt"])
        or not _SAFE_CODE_RE.fullmatch(value["outcome"])
    ):
        raise PipelineError("unsafe_generation_diagnostic")
    for key, item in value.items():
        if key in {"attempt", "outcome", "finish_reason"}:
            if not isinstance(item, str) or not _SAFE_CODE_RE.fullmatch(item):
                raise PipelineError("unsafe_generation_diagnostic")
        elif type(item) not in (int, float) or not math.isfinite(item):
            raise PipelineError("unsafe_generation_diagnostic")
    return value


def _split_output(output: object) -> tuple[str, dict]:
    if isinstance(output, str):
        return output, {}
    if not isinstance(output, dict) or not isinstance(output.get("text"), str):
        raise PipelineError("invalid_generator_output")
    return output["text"], _safe_telemetry(output)


def _evaluate(document: dict, output: object) -> tuple[dict | None, str | None, dict]:
    raw_response, telemetry = _split_output(output)
    try:
        payload = parse_model_output(raw_response)
    except EntitySchemaError as exc:
        return None, "schema:" + str(exc), telemetry
    grounded = ground_payload(document, payload)
    if not grounded["issues"]:
        return None, "grounding:all_empty", telemetry
    return grounded, None, telemetry


def _attempt_diagnostic(attempt: str, outcome: str, telemetry: dict | None = None) -> dict:
    return {
        "attempt": attempt,
        "outcome": outcome,
        **(telemetry or {}),
    }


def _decorate_entity(
    entity: dict,
    ordinal: int,
    contract_hash: str,
    attempt_diagnostics: list[dict],
) -> dict:
    return {
        **entity,
        "pipeline_version": PIPELINE_VERSION,
        "contract_hash": contract_hash,
        "document_ordinal": ordinal,
        "attempt_count": len(attempt_diagnostics),
        "generation_diagnostics": attempt_diagnostics,
    }


def _reject(
    document: dict,
    ordinal: int,
    contract_hash: str,
    error_codes: list[str],
    attempt_diagnostics: list[dict],
) -> dict:
    return {
        "schema_version": REJECT_SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "doc_id": document["doc_id"],
        "content_hash": document["content_hash"],
        "contract_hash": contract_hash,
        "document_ordinal": ordinal,
        "error_codes": error_codes,
        "attempt_count": len(attempt_diagnostics),
        "generation_diagnostics": attempt_diagnostics,
    }


def _load_terminal(
    run_dir: Path, contract_hash: str, known_documents: dict[str, str]
) -> dict[str, str]:
    terminal = {}
    for name, kind in ((ENTITIES_PRIVATE_NAME, "entity"), (REJECTS_PRIVATE_NAME, "reject")):
        path = run_dir / name
        if not path.exists():
            continue
        for _line_number, row in iter_jsonl(path):
            doc_id = row.get("doc_id")
            if doc_id not in known_documents:
                raise PipelineError("resume_unknown_terminal")
            if doc_id in terminal:
                raise PipelineError("resume_duplicate_terminal")
            if (
                row.get("content_hash") != known_documents[doc_id]
                or row.get("contract_hash") != contract_hash
            ):
                raise PipelineError("resume_terminal_contract_mismatch")
            terminal[doc_id] = kind
    return terminal


def _load_checkpoint(
    run_dir: Path,
    contract_hash: str,
    known_documents: dict[str, str],
    terminal_documents: set[str],
) -> dict[str, dict]:
    latest = {}
    observed_states = {}
    allowed_transitions = {
        None: {"primary_started"},
        "primary_started": {"needs_repair", "repair_started"},
        "needs_repair": {"repair_started"},
        "repair_started": set(),
    }
    path = run_dir / CHECKPOINT_PRIVATE_NAME
    if not path.exists():
        return latest
    for _line_number, event in iter_jsonl(path):
        doc_id = event.get("doc_id")
        if doc_id not in known_documents:
            raise PipelineError("resume_unknown_checkpoint")
        state = event.get("state")
        base_keys = {
            "pipeline_version",
            "doc_id",
            "content_hash",
            "contract_hash",
            "document_ordinal",
            "state",
        }
        expected_keys = (
            base_keys
            if state == "primary_started"
            else base_keys | {"primary_error_code", "primary_diagnostic"}
        )
        if (
            set(event) != expected_keys
            or event.get("pipeline_version") != PIPELINE_VERSION
            or event.get("content_hash") != known_documents[doc_id]
            or event.get("contract_hash") != contract_hash
            or type(event.get("document_ordinal")) is not int
            or event["document_ordinal"] < 0
            or state not in {"primary_started", "needs_repair", "repair_started"}
        ):
            raise PipelineError("resume_checkpoint_contract_mismatch")
        previous_state = observed_states.get(doc_id)
        if state not in allowed_transitions[previous_state]:
            raise PipelineError("resume_checkpoint_state_regression")
        observed_states[doc_id] = state
        if state != "primary_started":
            error_code = event.get("primary_error_code")
            if not isinstance(error_code, str) or not _SAFE_CODE_RE.fullmatch(error_code):
                raise PipelineError("resume_checkpoint_contract_mismatch")
            diagnostic = _validate_diagnostic(event.get("primary_diagnostic"))
            if diagnostic.get("attempt") != "primary":
                raise PipelineError("resume_checkpoint_contract_mismatch")
        if doc_id not in terminal_documents:
            latest[doc_id] = event
    return latest


def _rebuild_diagnostics(run_dir: Path) -> dict:
    target = run_dir / DIAGNOSTICS_SAFE_NAME
    temporary = target.with_name("." + target.name + ".tmp")
    terminal_count = 0
    attempt_count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for name, status in (
            (ENTITIES_PRIVATE_NAME, "entity"),
            (REJECTS_PRIVATE_NAME, "reject"),
        ):
            path = run_dir / name
            if not path.exists():
                continue
            for _line_number, terminal in iter_jsonl(path):
                terminal_count += 1
                diagnostics = terminal.get("generation_diagnostics")
                if not isinstance(diagnostics, list):
                    raise PipelineError("unsafe_generation_diagnostic")
                for diagnostic in diagnostics:
                    diagnostic = _validate_diagnostic(diagnostic)
                    safe_row = {
                        "schema_version": "generation_diagnostic_safe_v1",
                        "terminal_status": status,
                        **diagnostic,
                    }
                    output.write(canonical_json_bytes(safe_row).decode("utf-8") + "\n")
                    attempt_count += 1
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, target)
    return {"terminal_count": terminal_count, "generation_count": attempt_count}


@contextmanager
def _run_lock(run_dir: Path):
    try:
        import fcntl
    except ImportError as exc:
        raise PipelineError("linux_file_lock_required") from exc
    lock_path = Path(run_dir) / LOCK_PRIVATE_NAME
    try:
        handle = lock_path.open("a+", encoding="utf-8")
    except OSError as exc:
        raise PipelineError("run_lock_unavailable") from exc
    with handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PipelineError("extraction_already_running") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _extract_locked(
    run_dir: Path,
    config: dict,
    model_path: Path,
    *,
    generator=None,
    resume: bool = False,
) -> dict:
    run_dir = Path(run_dir)
    documents_path = run_dir / DOCUMENT_PRIVATE_NAME
    contract = build_run_contract(
        documents_path, config, Path(model_path), run_dir=run_dir
    )
    contract_path = run_dir / CONTRACT_PRIVATE_NAME
    output_paths = [
        run_dir / ENTITIES_PRIVATE_NAME,
        run_dir / REJECTS_PRIVATE_NAME,
        run_dir / CHECKPOINT_PRIVATE_NAME,
        run_dir / DIAGNOSTICS_SAFE_NAME,
    ]

    if resume:
        if not contract_path.is_file():
            raise PipelineError("resume_contract_missing")
        try:
            stored_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PipelineError("resume_contract_invalid") from exc
        if stored_contract != contract:
            raise PipelineError("resume_contract_mismatch")
        if any(not path.is_file() for path in output_paths[:3]):
            raise PipelineError("resume_journal_missing")
        for path in output_paths[:3]:
            _recover_trailing_partial(path)
    else:
        if contract_path.exists() or any(path.exists() for path in output_paths):
            raise PipelineError("fresh_extraction_outputs_exist")
        atomic_write_json(contract_path, contract)
        for path in output_paths[:3]:
            path.touch(exist_ok=False)

    known_documents = {}
    for _line_number, document in iter_jsonl(documents_path):
        known_documents[document["doc_id"]] = document["content_hash"]
    terminal = _load_terminal(run_dir, contract["contract_hash"], known_documents)
    checkpoint = _load_checkpoint(
        run_dir, contract["contract_hash"], known_documents, set(terminal)
    )
    diagnostics_before = _rebuild_diagnostics(run_dir)

    if len(terminal) == len(known_documents):
        return {
            "schema_version": "extraction_summary_safe_v1",
            "pipeline_version": PIPELINE_VERSION,
            "contract_hash": contract["contract_hash"],
            "documents_total": contract["documents"]["count"],
            "documents_processed_this_invocation": 0,
            "entities_written_this_invocation": 0,
            "rejects_written_this_invocation": 0,
            "repairs_started_this_invocation": 0,
            **diagnostics_before,
        }

    backend_holder = {"generator": generator}

    def configured_generator():
        if backend_holder["generator"] is None:
            from .vllm_backend import VLLMBackend

            backend_holder["generator"] = VLLMBackend(str(model_path), config)
        return backend_holder["generator"]

    entity_path = run_dir / ENTITIES_PRIVATE_NAME
    reject_path = run_dir / REJECTS_PRIVATE_NAME
    checkpoint_path = run_dir / CHECKPOINT_PRIVATE_NAME
    mode = "a"
    processed = 0
    accepted = 0
    rejected = 0
    repairs = 0

    with entity_path.open(mode, encoding="utf-8", newline="\n") as entity_file, reject_path.open(
        mode, encoding="utf-8", newline="\n"
    ) as reject_file, checkpoint_path.open(mode, encoding="utf-8", newline="\n") as checkpoint_file:

        def checkpoint_event(document: dict, ordinal: int, state: str, **values) -> None:
            event = {
                "pipeline_version": PIPELINE_VERSION,
                "doc_id": document["doc_id"],
                "content_hash": document["content_hash"],
                "contract_hash": contract["contract_hash"],
                "document_ordinal": ordinal,
                "state": state,
                **values,
            }
            _sync_jsonl(checkpoint_file, event)
            checkpoint[document["doc_id"]] = event

        def write_entity(row: dict) -> None:
            nonlocal processed, accepted
            _sync_jsonl(entity_file, row)
            terminal[row["doc_id"]] = "entity"
            processed += 1
            accepted += 1

        def write_reject(row: dict) -> None:
            nonlocal processed, rejected
            _sync_jsonl(reject_file, row)
            terminal[row["doc_id"]] = "reject"
            processed += 1
            rejected += 1

        def run_repair_batch(items: list[tuple[int, dict, str, dict]]) -> None:
            nonlocal repairs
            if not items:
                return
            for ordinal, document, primary_error, primary_diagnostic in items:
                checkpoint_event(
                    document,
                    ordinal,
                    "repair_started",
                    primary_error_code=primary_error,
                    primary_diagnostic=primary_diagnostic,
                )
            repairs += len(items)
            outputs = configured_generator().generate(
                [repair_prompt(document, config["max_input_chars"]) for _, document, _, _ in items],
                config["repair_max_tokens"],
            )
            if len(outputs) != len(items):
                raise PipelineError("repair_generation_count_mismatch")
            for (ordinal, document, primary_error, primary_diagnostic), output in zip(items, outputs):
                result, repair_error, telemetry = _evaluate(document, output)
                diagnostics = [
                    primary_diagnostic,
                    _attempt_diagnostic(
                        "repair", "entity" if result is not None else repair_error, telemetry
                    ),
                ]
                if result is not None:
                    write_entity(
                        _decorate_entity(
                            result, ordinal, contract["contract_hash"], diagnostics
                        )
                    )
                else:
                    write_reject(
                        _reject(
                            document,
                            ordinal,
                            contract["contract_hash"],
                            [primary_error, repair_error],
                            diagnostics,
                        )
                    )

        def run_primary_batch(items: list[tuple[int, dict]]) -> None:
            if not items:
                return
            for ordinal, document in items:
                checkpoint_event(document, ordinal, "primary_started")
            outputs = configured_generator().generate(
                [primary_prompt(document, config["max_input_chars"]) for _, document in items],
                config["primary_max_tokens"],
            )
            if len(outputs) != len(items):
                raise PipelineError("primary_generation_count_mismatch")
            repair_items = []
            for (ordinal, document), output in zip(items, outputs):
                result, primary_error, telemetry = _evaluate(document, output)
                diagnostic = _attempt_diagnostic(
                    "primary", "entity" if result is not None else primary_error, telemetry
                )
                if result is not None:
                    write_entity(
                        _decorate_entity(
                            result, ordinal, contract["contract_hash"], [diagnostic]
                        )
                    )
                else:
                    checkpoint_event(
                        document,
                        ordinal,
                        "needs_repair",
                        primary_error_code=primary_error,
                        primary_diagnostic=diagnostic,
                    )
                    repair_items.append((ordinal, document, primary_error, diagnostic))
            run_repair_batch(repair_items)

        primary_buffer: list[tuple[int, dict]] = []
        repair_buffer: list[tuple[int, dict, str, dict]] = []
        for ordinal, (_line_number, document) in enumerate(iter_jsonl(documents_path)):
            doc_id = document["doc_id"]
            if doc_id in terminal:
                continue
            state = checkpoint.get(doc_id)
            if state and state["document_ordinal"] != ordinal:
                raise PipelineError("resume_checkpoint_ordinal_mismatch")
            if state and state["state"] == "repair_started":
                primary_error = state.get("primary_error_code", "primary:unknown")
                primary_diagnostic = state.get(
                    "primary_diagnostic",
                    _attempt_diagnostic("primary", primary_error),
                )
                diagnostics = [
                    primary_diagnostic,
                    _attempt_diagnostic("repair", "repair:interrupted"),
                ]
                write_reject(
                    _reject(
                        document,
                        ordinal,
                        contract["contract_hash"],
                        [primary_error, "repair:interrupted"],
                        diagnostics,
                    )
                )
                continue
            if state and state["state"] in {"primary_started", "needs_repair"}:
                primary_error = state.get("primary_error_code", "primary:interrupted")
                primary_diagnostic = state.get(
                    "primary_diagnostic",
                    _attempt_diagnostic("primary", primary_error),
                )
                repair_buffer.append((ordinal, document, primary_error, primary_diagnostic))
                if len(repair_buffer) >= config["batch_size"]:
                    run_repair_batch(repair_buffer)
                    repair_buffer = []
                continue
            primary_buffer.append((ordinal, document))
            if len(primary_buffer) >= config["batch_size"]:
                run_primary_batch(primary_buffer)
                primary_buffer = []

        run_repair_batch(repair_buffer)
        run_primary_batch(primary_buffer)

    diagnostics = _rebuild_diagnostics(run_dir)
    return {
        "schema_version": "extraction_summary_safe_v1",
        "pipeline_version": PIPELINE_VERSION,
        "contract_hash": contract["contract_hash"],
        "documents_total": contract["documents"]["count"],
        "documents_processed_this_invocation": processed,
        "entities_written_this_invocation": accepted,
        "rejects_written_this_invocation": rejected,
        "repairs_started_this_invocation": repairs,
        **diagnostics,
    }


def extract(
    run_dir: Path,
    config: dict,
    model_path: Path,
    *,
    generator=None,
    resume: bool = False,
) -> dict:
    run_dir = Path(run_dir)
    with _run_lock(run_dir):
        return _extract_locked(
            run_dir,
            config,
            model_path,
            generator=generator,
            resume=resume,
        )
