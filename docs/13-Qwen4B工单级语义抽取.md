# 13-Qwen3-4B 工单级语义抽取

## 1. 目标与边界

本流程对每条已脱敏 12345 工单执行一次 Qwen3-4B 主调用，同时生成语义完整 event、开放式问题对象/行为、道路/路口/POI，以及 intent、emotion、satisfaction、urgency。只有确定性校验为 `repair_required` 的工单允许一次 repair 调用。

本地只开发、打包和运行不加载模型的单元测试。模型下载、真实脱敏数据、GPU 推理、性能和质量验收全部在服务器进行。不得把模型权重、工单 JSONL、响应、rejects、links、DuckDB 或日志提交 GitHub。

## 2. 服务器准备

```bash
nvidia-smi
df -h
pip install -r requirements.sag.txt
pip install -r requirements.entity.txt
```

PyTorch 必须按服务器 CUDA 环境单独安装，不在基础 requirements 中固定。默认 Transformers 后端继续使用现有 `ragflow-embed` 环境。vLLM wheel 与特定 PyTorch/CUDA/XFormers 二进制栈绑定，**不要 clone `ragflow-embed`**，也不要在原环境上强行覆盖 Torch；使用全新 Python 3.11 环境并安装固定的 `vllm==0.8.5`：

```bash
conda create -n sag-vllm python=3.11 -y
conda activate sag-vllm
python -m pip install --upgrade pip setuptools wheel
python -m pip install --timeout 120 --retries 10 --prefer-binary \
  --only-binary=vllm -r requirements.vllm.txt
python -m pip check
python - <<'PY'
import torch, vllm
print({
    "torch": torch.__version__, "vllm": vllm.__version__,
    "cuda": torch.version.cuda, "available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
    "capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
})
PY
```

`--only-binary=vllm` 可阻止 pip 静默转为耗时且易失败的 vLLM 源码编译。项目同时固定 `transformers==4.51.3`，避免 vLLM 0.8.5 的宽松下限解析到不兼容的 Transformers 5.x。若镜像出现 `ReadTimeoutError`，不要删除已经缓存的 326MB vLLM wheel；用命令行 `--index-url` 切换可访问的 PyPI 镜像，并保留 `--timeout 120 --retries 10`。若安装失败，先保存 pip 输出末尾，不要在同一半安装环境反复强装；必要时删除并重建 `sag-vllm`，不影响 Transformers 回退。V100 路径会在导入 vLLM 前默认设置 `VLLM_USE_V1=0` 和 `VLLM_ATTENTION_BACKEND=XFORMERS`；用户显式设置的环境变量优先。模型目录默认：

```text
models/Qwen3-4B
```

生产输入必须是脱敏多视图 JSONL，默认：

```text
data/t_order_master.100k.multiview.jsonl
```

每行至少包含稳定 `doc_id`、`case_content_clean`，并可包含 `title_clean`、`case_goal_clean`、`address_detail_clean`、`metadata`。不要把原始 TSV 交给新语义抽取入口。

## 3. 上传交付包

Windows 本地打包：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/package_entity_extraction.ps1
```

把 `packages/sag-qwen3-4b-semantic-extraction.zip` 上传服务器，校验输出的 SHA-256 后解压。包中只包含源码、测试、配置、脚本、本文档和 requirements。

## 4. 10 条 smoke

```bash
BATCH_SIZE=1 LIMIT=10 bash scripts/extract_semantics_qwen3_4b.sh
python scripts/check_semantic_run.py \
  --semantic outputs/work_order_semantics.qwen3_4b.jsonl \
  --rejects outputs/work_order_semantics.rejects.jsonl \
  --run-report outputs/work_order_semantics.run.json \
  --quality-report outputs/work_order_semantics.quality.json
```

检查：无 OOM；processed 数量正确；每工单一次 primary；repair 不超过 repair-required 数量；`finish_reason=length` 可解释；报告中不含 prompt、正文、evidence 或原始响应。当前 `sag_semantic_v7` 的 primary 上限为 640 tokens，只有整单 repair 使用 768 tokens。v7 会对候选级 sanitation 做最多三轮确定性复核，并仅在字符串 entity 逐字存在于 clean field 且满足 road/intersection/POI 类型 gate 时恢复。两条在同一 clean field 中分别验证的 road 能定位到严格路口原文时可确定性合成 intersection；裸行政区和纯门牌不作为 POI；咨询/办理/注册/就学等正常服务动作不作为 behavior；高风险 canonical 状态必须得到 evidence 支持；intent 只在没有可靠结果时补一个主意图；emotion/satisfaction 必须有直接对应表达。`semantic_gap:*` 只作审计，不触发模型 repair。完整顶层对象仅允许 trailing comma、JSON 控制字符和纯 Python 字面量三种安全 parser 恢复；截断对象、空 event 和历史污染仍必须通过原有一次 repair，不能确定性猜测。

每次运行默认同时生成隐私安全诊断日志：

```text
outputs/work_order_semantics.qwen3_4b.jsonl.diagnostics.jsonl
```

可通过 `DIAGNOSTIC_LOG=outputs/smoke10.diagnostics.jsonl` 自定义路径。日志只记录哈希化的 `order_ref`、阶段、token、耗时、finish reason、验证错误码、候选数量、清洗动作和 GPU 当前/峰值，不记录 `doc_id`、工单正文、prompt、evidence 或模型原始响应。模型加载或调用失败时只记录异常类型，不记录可能含正文的异常消息。

运行后生成可直接回传的安全汇总：

```bash
python scripts/summarize_semantic_diagnostics.py \
  --input outputs/work_order_semantics.qwen3_4b.jsonl.diagnostics.jsonl \
  > outputs/work_order_semantics.diagnostics.summary.json
```

优先回传该 summary、run report、quality report 和 checker 输出。需要分析逐工单错误码或显存随批次变化时，再回传 diagnostics JSONL；两者都不得提交 GitHub。

## 5. 995 条样本

```bash
LIMIT=995 bash scripts/extract_semantics_qwen3_4b.sh
bash scripts/project_semantics_to_sag.sh
LIMIT=995 bash scripts/build_sag_semantic_100k.sh
```

人工抽样必须覆盖：开放领域主题、诉求动作/问题行为、road/intersection/POI、历史答复/当前立场、模板谢谢、对象态度/诉求人情绪和 discourse。验证器通过率不能当作准确率。重点比较 quality report 中 `all_entities_empty_rate`、`intent_coverage`、`repair_attempted_count`、`json_recovery_count`、`semantic_gap_counts` 及 warning counts；字符串 entity 和合成 intersection 必须同时满足逐字 evidence 和类型 gate。面向 SAG 的正式验收、生产分布/挑战集抽样、私有 candidate ledger、Issue 标注及 Oracle flat/issue-aware 超边比较见 `docs/14-SAG语义评测与Issue标注.md`；实体非空 coverage 不能替代 mention、attachment、false co-membership 和最终检索指标。

## 6. 100k 与恢复

batch 1 仅用于早期调试，不满足生产吞吐。smoke 正确性和显存通过后，使用全新输出目录先测 `BATCH_SIZE=8`；若 GPU 峰值低于约 28GB 且无 OOM，再比较 4/8/16：

```bash
LIMIT=32 BATCH_SIZES="4 8 16" bash scripts/benchmark_semantic_batches.sh
```

脚本为每个 batch size 启动独立进程，避免前一轮 CUDA cache 干扰后一轮；比较 `orders_per_second`、`output_tokens_per_second`、reject/repair 和 GPU 峰值，以质量不下降条件下吞吐最高者作为 995 配置。Transformers 路径使用 SDPA 和 KV cache，批次间默认不调用 `empty_cache()`，以复用 allocator；应依据 current allocated 判断真实张量是否增长，reserved 增长本身不等于泄漏。

Transformers batch 8 的 32 条实测为 32/32 records、0 reject、1 repair、0 truncation、`0.1895 orders/s`、峰值 allocated/reserved `10.387/12.572GB`。相对 batch 1 提升约 2.9 倍，但 100k 仍约需 6.1 天，因此不能作为最终生产配置。先用剩余显存测试 Transformers 16/32：

```bash
BACKEND=transformers LIMIT=32 BATCH_SIZES="16 32" \
  bash scripts/benchmark_semantic_batches.sh
```

随后在隔离的 `sag-vllm` 环境测试 offline continuous batching 和 paged KV cache。V100 + Torch 2.6/Triton 3.2 的 prefix-prefill 内核可能直接触发 `LLVM ERROR: Failed to compute parent layout for slice layout`，因此该硬件路径默认关闭 prefix caching 和 chunked prefill。vLLM 的 batch size 是一次提交给调度器的工单数；由于输出长度差异很大，建议先测 32，再用 50 条 smoke 测 50，995 阶段再测 64：

```bash
conda activate sag-vllm
BACKEND=vllm LIMIT=32 BATCH_SIZES="32" \
  bash scripts/benchmark_semantic_batches.sh
BACKEND=vllm LIMIT=50 BATCH_SIZES="50" \
  bash scripts/benchmark_semantic_batches.sh
```

默认 vLLM 配置为 FP16、`gpu_memory_utilization=0.85`、`max_model_len=4096`、`max_num_seqs=64`、prefix caching=false、chunked prefill=false，并保留 CUDA graph。V100 batch 32 的 v5 smoke 已达到约 `1.56 orders/s` primary 稳态吞吐；v7 保留跨 primary batch 的 repair 聚合，`repair_batch_size` 默认 8，避免 singleton repair 成为主要耗时。每条工单仍最多一次 repair；报告用 `primary_requests/repair_requests` 表示工单请求条数，用 `primary_batches/repair_batches` 表示实际模型批次数。可通过 `REPAIR_BATCH_SIZE=16` 独立调优。若关闭 prefix caching 后仍出现同一 LLVM 错误，第二级回退设置 `VLLM_ENFORCE_EAGER=1`；eager 会降低吞吐，只用于确认是否为剩余编译路径。发生 OOM 时先设置 `VLLM_GPU_MEMORY_UTILIZATION=0.75` 或降低 batch；启动时提示上下文不足才增加 `VLLM_MAX_MODEL_LEN`，不要盲目增大。后端只改变推理执行，不改变工单级 schema 和投影；checkpoint 身份仍为 `(doc_id, content_hash, prompt_version, model_id)`。benchmark/smoke 必须使用全新目录，不设置 `RESUME=1`。995 样本质量通过后运行：

```bash
LIMIT=100000 bash scripts/extract_semantics_qwen3_4b.sh
bash scripts/project_semantics_to_sag.sh
LIMIT=100000 bash scripts/build_sag_semantic_100k.sh
```

中断恢复：

```bash
RESUME=1 LIMIT=100000 bash scripts/extract_semantics_qwen3_4b.sh
```

指定 doc_id 重跑：

```bash
DOC_ID_FILE=retry_doc_ids.txt RETRY_REJECTED=1 LIMIT=100000 \
  bash scripts/extract_semantics_qwen3_4b.sh
```

checkpoint 身份为 `(doc_id, content_hash, prompt_version, model_id)`。正文变化或 Prompt/模型变化不会被旧 checkpoint 错误跳过。同一模型和 Prompt 在 Transformers/vLLM 间切换时结果可恢复复用；若目的是对比后端，必须使用不同的新输出目录且不设置 `RESUME=1`。

## 7. 产物

```text
outputs/work_order_semantics.qwen3_4b.jsonl
outputs/work_order_semantics.rejects.jsonl
outputs/work_order_semantics.run.json
outputs/work_order_semantics.quality.json
outputs/work_order_semantics.qwen3_4b.jsonl.diagnostics.jsonl
outputs/work_order_semantics.diagnostics.summary.json
outputs/sag_events.qwen3_4b.jsonl
outputs/sag_event_entity_links.qwen3_4b.jsonl
outputs/sag_event_discourse.qwen3_4b.jsonl
outputs/sag_semantic.qwen3_4b.100k.duckdb
```

工单级 semantics JSONL 是权威审计中间产物；links/discourse/event rows 是可重复生成的 SAG 投影。不要用 event summary 替代完整脱敏 chunk。

## 8. 服务器性能记录

保留但不要提交：backend、GPU 型号、dtype、batch size、attention/cache implementation、prefix caching、input/output token 总量及 p50/p95、finish reason、repair/reject/OOM、elapsed seconds、orders/s、output tokens/s、GPU 利用率和 current/peak allocated/reserved。长短文本按配置分桶；短 smoke 的 orders/s 包含模型加载开销，batch benchmark 应至少运行 32 条。Transformers 与 vLLM 必须使用相同输入、Prompt 版本和 token 上限比较，不能改变工单级 schema 和投影。

## 9. 故障恢复

- Transformers OOM：降低 `batch_size`，从 checkpoint `RESUME=1`；不要删除已完成 partial。
- vLLM 安装失败：确认 Linux x86_64、Python 3.9–3.12（推荐 3.11）、pip 可看到 `vllm==0.8.5` wheel、磁盘空间充足；不要 clone 原环境。resolver 报 torch/xformers/triton 冲突时删除并重建全新环境，不使用 `--no-deps`。
- vLLM 启动失败：保留完整 console 日志；V100 必须使用 V0/XFormers/FP16。若首次生成出现 `Failed to compute parent layout for slice layout`，确认日志为 `enable_prefix_caching=False`、`chunked_prefill_enabled=False`；仍失败再设置 `VLLM_ENFORCE_EAGER=1`。不要改为 FP32：4B 权重和 KV cache 显存会显著增加且吞吐更差。若仍失败，切回 `BACKEND=transformers`，不要修改原 `ragflow-embed` 环境。
- vLLM OOM：先将 `VLLM_GPU_MEMORY_UTILIZATION` 从 0.85 降至 0.75，再降低 batch；paged KV cache 会预留显存，不能只用 PyTorch allocated 判断整卡占用。
- 大量 `length`：先确认 diagnostics 中 primary/repair 的 `max_new_tokens` 分别为 640/768；仍有截断时再调整配置并升级 `prompt_version`，不要直接复用旧 checkpoint。
- 大量 repair：用 `summarize_semantic_diagnostics.py` 对比 `validation_before`、候选级 sanitation 和 `validation_after`，不要默认所有工单双调用。
- OOM 或调用异常：查看 diagnostics 最后一条是 `run_failed:model_load` 还是 `model_call_failed`，并结合 `batch_memory` 的 current/peak allocated/reserved 区分模型常驻张量、临时峰值和 PyTorch reserved cache。
- ID 对不上：确认 input、semantics、projection 使用同一稳定 `doc_id/content_hash`。
- 只需重投影：直接运行 `project_semantics_to_sag.sh` 和 `build_sag_semantic_100k.sh`，不加载模型。

回传时优先返回 checker 输出、run/quality 报告和哈希；原始响应或含正文产物只能通过受控安全通道传输。

## 10. v8_dev1 最小 Issue Schema 与服务器循环

`sag_semantic_v8_dev1` 是与冻结 v7 并存的开发契约，不覆盖 v7 parser、validator 或 replay。模型输出版本为 `sag_semantic_issue_output_v1`，核心结构为：

```json
{
  "event_summary": "",
  "issues": [{
    "time_scope": "current",
    "objects": [],
    "problem_behaviors": [],
    "question_focus": [],
    "request_actions": [],
    "locations": []
  }],
  "discourse": {
    "intents": [],
    "emotions": [],
    "satisfaction": {"label":"unknown","target":"","field":"","evidence":""},
    "urgency": {"level":"normal","field":"","evidence":""}
  }
}
```

一个 issue 是一个现实业务关注点，不是一个话语动作。问题事实及针对它的诉求属于同一 issue；纯咨询使用 `question_focus`，不制造 problem behavior；只有合并后会产生错误对象—行为、对象—地点或动作—对象关系时才拆 issue。模型不输出 `canonical/confidence/issue_id`。surface/evidence 是事实权威；canonical/alias 留给独立、可撤销的 normalization 层。

v8 版本独立记录：

- Prompt：`sag_semantic_v8_dev1`
- 输出 schema：`sag_semantic_issue_output_v1`
- validator：`sag_semantic_issue_validator_v1`
- projection：`sag_semantic_issue_projection_v1`
- decoder：`unconstrained_json_v1`

每个 issue 确定性投影为独立 SAG event。`problem_object/problem_behavior/road/intersection/poi` 可作为默认 expansion frontier；`issue_predicate/request_action` 只作查询属性成员，不作为默认 frontier。可靠 metadata 复制到每个 issue；正文 regex 规则不会复制到所有 issue，避免把 issue-aware 投影重新污染为 flat。

### 10.1 首轮服务器 smoke

先在本地或服务器私有目录将冻结的 48 条 manifest 确定性拆成 16 条 development 与 32 条 holdout。该工具只读取 identity/stratum，不读取正文，safe report 不含 `doc_id/content_hash`：

```bash
PYTHONPATH=src python scripts/split_semantic_eval_manifest.py \
  --source private/eval.pilot.manifest.private.jsonl \
  --development private/eval.v8.development-16.manifest.private.jsonl \
  --holdout private/eval.v8.holdout-32.manifest.private.jsonl \
  --report private/eval.v8.split.safe.json \
  --dev-size 16 --seed sag-v8-split-v1
```

只使用脱敏 inference packet 和 development manifest。`LIMIT=16` 在 exact identity 模式下不会代替 manifest split；不要把 48 条总 manifest 当作首轮 16 条。不要传原始 TSV，不连接其他数据：

```bash
conda activate sag-vllm
export INPUT_JSONL=private/eval.pilot.inference.private.jsonl
export IDENTITY_MANIFEST=private/eval.v8.development-16.manifest.private.jsonl
export MODEL_PATH=models/Qwen3-4B
export BACKEND=vllm
export RUN_DIR=outputs/v8-dev1-smoke-001
bash scripts/extract_semantics_v8_dev1.sh
```

入口要求显式 `INPUT_JSONL/IDENTITY_MANIFEST/MODEL_PATH`，并拒绝 `RESUME=1` 和已存在 `RUN_DIR`，防止误跑、v7/v8 或不同 Prompt 结果混用。V100 固定：

```bash
VLLM_USE_V1=0
VLLM_ATTENTION_BACKEND=XFORMERS
VLLM_ENABLE_PREFIX_CACHING=0
VLLM_ENABLE_CHUNKED_PREFILL=0
SEMANTIC_LLM_DTYPE=float16
```

v8 默认 `max_model_len=8192`、`max_num_seqs=32`，其 KV 上限与 v7 的 `4096×64` 同量级；primary `max_new_tokens=1024`、整单 repair=768。先用 batch 8；看到真实 OOM 后再降 `gpu_memory_utilization` 或 batch，不预先牺牲吞吐。只有关闭 prefix/chunked 后仍出现相同 LLVM slice layout 错误，才试 `VLLM_ENFORCE_EAGER=1`。

### 10.2 首轮必须回传的安全结果

只回传或汇报：

- `$RUN_DIR/check.safe.json`
- `$RUN_DIR/run.safe.json`
- `$RUN_DIR/quality.safe.json`
- diagnostics 聚合 summary
- semantic/reject/candidate/decision 文件 SHA-256

不要粘贴 semantic、reject、candidate、decision、Prompt、evidence 或正文。`quality.safe.json` 重点观察：

- `issue_count_distribution`
- `issue_role_coverage`
- `status_counts/warning_counts`
- repair/reject/truncation
- primary/repair output token p50/p95
- latency、orders/s、output tokens/s、GPU peak

第一轮先跑冻结的 16 条 development；后续 `v8_dev2/dev3` 仍复用这 16 条，不得换 seed 重抽。根据实际 Qwen 错误迭代，不要看到单例错误就加关键词。Prompt 冻结为 RC 后才首次打开 32 条 holdout；之后再扩大到 48/100、995 和 100k。v7 与 v8 对比必须使用相同 identity、模型和输入；只改变 schema/Prompt，并保持输出目录隔离。

### 10.3 v8_dev1 development 结果与 v8_dev2

首轮冻结 16 条 development 的安全聚合结果为：5 records、11 rejects、11 repairs、0 truncation，27 次 generation 均为 `finish_reason=stop`。主要错误不是 JSON、显存或 token，而是模型将 issue 的 `objects/problem_behaviors/question_focus/request_actions` 大量输出为字符串数组，导致 `malformed_issue_member` 和 `empty_issue`；repair 仍重复该结构。地点及 discourse 的不可靠 evidence 已由候选级 sanitation 删除，不是主要 reject 根因。

`v8_dev2` 保持 `sag_semantic_issue_output_v1`、projection v1、decoder、token、V100 参数和冻结 development 不变，只做两项版本化修复：

- Prompt 增加完整非空 member/location/discourse 对象格式，明确禁止 `["路灯"]` 式字符串数组；
- validator 升为 `sag_semantic_issue_validator_v2`。字符串 issue member 只作为未信任候选保留，且仅当 surface 逐字存在于四个 clean fields 时确定性补 `field/evidence`；无法验证的候选仍删除。错误 location grounding 也只允许从逐字 surface 恢复。

第二轮继续使用相同 `INPUT_JSONL`、相同 development manifest 和全新目录：

```bash
export BACKEND=vllm
export RUN_DIR=/path/outside/repo/v8-dev2-smoke-001
bash scripts/extract_semantics_v8_dev2.sh
```

不得对 v8_dev1 使用 resume，也不得打开 holdout。对比重点是 records/rejects/repairs、`malformed_issue_member`、`empty_issue`、grounded issue role coverage 以及是否出现新的过度恢复；validator 通过率仍不能当作语义 precision/recall。
