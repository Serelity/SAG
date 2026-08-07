# SAG / Qwen3-4B 12345 工单语义抽取交接文档

> 写给完全没有历史上下文的新会话。
>
> 最后更新：2026-08-08。
>
> 本文只记录代码、版本、聚合指标、安全哈希和私有文件路径；不含工单正文、evidence、`doc_id`、原始模型响应或人工标注内容。

---

## 0. 新会话第一屏：当前状态和唯一下一步

### 当前代码

- 仓库：`G:\RAG\SAG`
- 功能工作树：`G:\RAG\SAG\.worktrees\qwen4b-semantic-extraction`
- 分支：`feature/qwen4b-semantic-extraction`
- v8_dev2 实现基线提交（当前 HEAD 会因本交接文档提交而更新，但必须包含此提交）：

  ```text
  af665198adfa3d12bfc6daedc3f0244d3d8a84f0
  fix: ground issue member strings in v8 dev2
  ```

- 前一提交：

  ```text
  8cfc408bd09f45c330f41a23d3b226c953a5892a
  feat: add issue-aware semantic extraction v8
  ```

- `origin/main` 仍可能停在 `834b259`；不要误以为 dev2 已合并 main。当前工作只在功能分支。
- 当前源码工作树在本交接更新前是干净的；更新本文件后应提交并推送。

### 已完成的真实 GPU 结果

冻结的 16 条 development 已在服务器用 Qwen3-4B / vLLM / V100 完成 `sag_semantic_v8_dev1`：

- 16 条输入，精确 manifest 命中；
- 5 records；
- 11 rejects；
- 11 repairs；
- 0 truncation；
- 27 次 generation 全部 `finish_reason=stop`；
- output throughput `160.225 tokens/s`；
- GPU peak allocated `26.888 GB`；
- GPU peak reserved `28.189 GB`；
- candidate/decision ledger 各 27 条，已完整落盘。

根因已经由安全聚合报告定位：**不是 JSON、token、显存、CUDA、vLLM 或截断问题**。Qwen 大量把 issue member 输出成字符串数组，例如概念上等价于 `objects=["..."]`，而契约要求每项是 `{surface,field,evidence}` 对象；repair Prompt 没展示完整非空对象，所以 repair 继续重复同类错误，产生大量 `malformed_issue_member` 和 `empty_issue`。

### 已完成的 dev2 修复

`v8_dev2` 已实现、测试、提交和推送：

- Prompt：`sag_semantic_v8_dev2`
- output schema：仍为 `sag_semantic_issue_output_v1`
- validator：`sag_semantic_issue_validator_v2`
- projection：仍为 `sag_semantic_issue_projection_v1`
- decoder：仍为 `unconstrained_json_v1`
- primary token：仍为 1024
- repair token：仍为 768
- V100/vLLM 参数不变
- Prompt 增加完整非空 member/location/discourse 对象示例，明确禁止字符串数组；
- validator v2 只把字符串 member 当作未信任候选；只有 surface 逐字存在于四个 clean fields 时才确定性补 `field/evidence`；无法验证仍删除；
- location 的错误 field/evidence 也只能通过逐字 surface 恢复；
- dev1 Prompt 与 validator v1 路径保持冻结，可重放。

### 当前唯一下一步

**在同一冻结 16 条 development 上，用全新目录运行 `v8_dev2`。**

不要：

- 打开 32 条 holdout；
- 运行 48、100、995 或 100k；
- 使用 `RESUME=1`；
- 复用 dev1 的 `RUN_DIR`；
- 提高 token；
- 修改 CUDA/vLLM；
- 重抽 development；
- 查看或粘贴私有正文、evidence 或原始响应。

服务器无法访问外网。若不能 `git pull`，上传下述 Linux 友好离线包：

```text
G:\RAG\_tmp_sag_package_v8_dev2_final\sag-qwen3-4b-semantic-extraction-af66519.tar.gz
sha256:0b7169b42fbf9dea60e561cf34ca1acac310d816f6b197fe8aebe76d04fa5eb2
```

该包：

- `package_commit=af665198adfa3d12bfc6daedc3f0244d3d8a84f0`
- 126 个条目；
- manifest 缺失/多余/哈希不一致均为 0；
- 私有或禁止条目为 0；
- Linux 解包后 212 项测试通过。

原来服务器上的两个小私有文件不变，不需要重新上传：

```text
eval.pilot.inference.private.jsonl                 # 48 条脱敏输入
eval.v8.development-16.manifest.private.jsonl      # 16 条冻结 development identity
```

---

## 1. 我们在做什么

目标是把任意领域的中文 12345 工单转换成可审计、可重放、适合 SAG 检索的联合语义结构。

不是普通 NER。每张工单由 Qwen3-4B 一次联合识别：

- 当前事件概括；
- 一个或多个现实业务 issue；
- 每个 issue 的对象；
- 已发生问题/异常行为；
- 咨询焦点；
- 诉求动作；
- road / intersection / POI；
- 工单级 intent、emotion、satisfaction、urgency。

核心执行约束：

1. 每工单只能有一次 primary generation；
2. 只有整单异常最多一次 repair；
3. 多行 SAG link 是 Python 投影粒度，不能让模型按实体重复读取工单；
4. 工单级脱敏 semantic JSONL 是权威中间产物；
5. SAG event/entity/discourse 表是确定性、可重放投影；
6. 新生产入口只消费脱敏 multiview/inference JSONL，原 TSV 不能直接交给模型；
7. 保留完整脱敏 chunk，event summary 不能替代事实源；
8. 开放域识别，不用固定业务词表限制主题；
9. metadata 由 Python 处理，不要求 Qwen 重复输出；
10. discourse 默认不是 SAG expansion frontier，避免超级节点污染。

质量分三层：

1. grounded semantic extraction；
2. issue/hyperedge 关系纯度；
3. SAG Precision@K、Recall@K、nDCG、MRR 和错误扩展率。

当前只处于第 1–2 层的开发阶段，不能因为 validator 通过率提高就宣布语义准确率通过。

---

## 2. 当前最小 issue 契约

一个 issue 表示一个**现实业务关注点**，不是句子、话语动作或单一标签。

模型输出：

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

定义：

- problem、question、request 是 issue 内不同角色；
- 问题事实与针对它的诉求默认属于同一 issue；
- 纯咨询用 `question_focus`，不能制造 problem behavior；
- 历史答复和当前仍未解决状态通常是同一现实关注点的上下文；
- 只有合并会制造原文没有的对象—行为、对象—地点或动作—对象关系时才拆 issue；
- 不使用单一 `mode`；各角色数组是否为空表达组合；
- 模型不输出 `canonical`、`confidence`、`issue_id`、metadata 或数据库 ID；
- Python 按数组顺序生成 issue event ID；
- 每个 issue 投影为独立 SAG event；
- `problem_object/problem_behavior/road/intersection/poi` 可进入默认 frontier；
- `issue_predicate/request_action` 不进入默认 expansion frontier。

三层 schema 必须分开：

1. Qwen 输出 schema；
2. Python 权威 semantic record；
3. Python SAG projection。

正式技术 schema：

```text
configs/sag_semantic_issue_output_v1.schema.json
```

避免目标 decoder 子集不一定支持的关键字：`maxItems/minItems/maxLength/uniqueItems` 等不在正式 schema 中；数量上限由 Prompt/validator 管理。

---

## 3. 版本和关键文件

### v7 冻结基线

- Prompt：`sag_semantic_v7`
- validator：`sag_semantic_validator_v1`
- projection：`sag_semantic_projection_v1`
- config：`configs/sag_semantic_extraction_qwen3_4b.json`
- primary/repair：640/768

除非明确开展 v7 新版本，不要改：

```text
src/ragflow_style_pipeline/sag_semantic_prompt.py
src/ragflow_style_pipeline/sag_semantic_schema.py
src/ragflow_style_pipeline/sag_semantic_validation.py
configs/sag_semantic_extraction_qwen3_4b.json
tests/test_sag_semantic_prompt.py
tests/test_sag_semantic_schema.py
tests/test_sag_semantic_validation.py
```

### v8_dev1 基线

- commit：`8cfc408`
- Prompt：`sag_semantic_v8_dev1`
- validator：`sag_semantic_issue_validator_v1`
- config：`configs/sag_semantic_extraction_qwen3_4b_v8_dev1.json`
- entry：`scripts/extract_semantics_v8_dev1.sh`

保留 dev1 结果和代码，用于对比；不要覆盖或 resume。

### v8_dev2 当前版本

- commit：`af66519`
- Prompt：`sag_semantic_v8_dev2`
- validator：`sag_semantic_issue_validator_v2`
- output schema：`sag_semantic_issue_output_v1`
- projection：`sag_semantic_issue_projection_v1`
- config：`configs/sag_semantic_extraction_qwen3_4b_v8_dev2.json`
- entry：`scripts/extract_semantics_v8_dev2.sh`

关键实现：

```text
src/ragflow_style_pipeline/sag_semantic_issue_prompt.py
src/ragflow_style_pipeline/sag_semantic_issue_schema.py
src/ragflow_style_pipeline/sag_semantic_issue_validation.py
src/ragflow_style_pipeline/sag_semantic_llm.py
src/ragflow_style_pipeline/sag_semantic_projection.py
src/ragflow_style_pipeline/sag_semantic_audit.py
src/ragflow_style_pipeline/sag_db.py
src/ragflow_style_pipeline/sag_semantic_versions.py
```

关键测试：

```text
tests/test_sag_semantic_issue.py
tests/test_sag_semantic_issue_pipeline.py
tests/test_sag_semantic_llm.py
tests/test_sag_semantic_audit.py
tests/test_sag_db.py
tests/test_check_semantic_run.py
tests/test_split_semantic_eval_manifest.py
```

---

## 4. 数据、development/holdout 和私有文件

### 本地私有根

```text
G:\RAG\SAG_private\semantic-eval\current-v2\
```

权威输入：

```text
G:\RAG\SAG_private\semantic-eval\current-v2\input\t_order_master.100k.multiview.private.jsonl
```

最小 48 条模型输入：

```text
G:\RAG\SAG_private\semantic-eval\current-v2\model-input\eval.pilot.inference.private.jsonl
```

安全属性：

- 48 条；
- 48,295 bytes；
- SHA-256：`sha256:75b902a8e494faa81cbc41bfb3cc5f14397cc2832436ce492d9cfd7ffa49811a`。

冻结 development manifest：

```text
G:\RAG\SAG_private\semantic-eval\current-v2\audit\eval.v8.development-16.manifest.private.jsonl
```

- 16 条；
- production 8 / challenge 8；
- SHA-256：`sha256:7bfe205abc66efb59c44692b9f6bba57ad70826f52eed643e1435dc237e62840`。

冻结 holdout manifest：

```text
G:\RAG\SAG_private\semantic-eval\current-v2\audit\eval.v8.holdout-32.manifest.private.jsonl
```

- 32 条；
- production 16 / challenge 16；
- SHA-256：`sha256:d3dc039cbbeae13433c72253bfddc6dab3fdf8e8e07eecc340c95e7e728b460b`。

split：

- seed：`sag-v8-split-v1`
- identity overlap：0
- safe report：

  ```text
  G:\RAG\SAG_private\semantic-eval\current-v2\audit\eval.v8.split.safe.json
  ```

**所有 dev 版本必须复用同一16条。Prompt 冻结为 RC 后才首次打开32条 holdout。**

### 服务器现有私有输入

用户已把以下两个文件放到服务器私有目录，dev2 不需重新上传：

```text
/seu_share/home/huangkai/220243809/RAG/SAG_private/v8-smoke/eval.pilot.inference.private.jsonl
/seu_share/home/huangkai/220243809/RAG/SAG_private/v8-smoke/eval.v8.development-16.manifest.private.jsonl
```

历史 dev1 run：

```text
/seu_share/home/huangkai/220243809/RAG/SAG_private/v8-smoke/run-001/
```

不要删除、覆盖或 resume。它是有效基线。

服务器不能访问外网。不要要求服务器访问 GitHub、Hugging Face 或 PyPI；代码用离线包，模型和 Conda 环境用服务器既有资源。

---

## 5. v8_dev2 的服务器运行方式

### 5.1 离线部署代码

优先上传：

```text
G:\RAG\_tmp_sag_package_v8_dev2_final\sag-qwen3-4b-semantic-extraction-af66519.tar.gz
```

校验：

```bash
sha256sum /上传位置/sag-qwen3-4b-semantic-extraction-af66519.tar.gz
```

应为：

```text
0b7169b42fbf9dea60e561cf34ca1acac310d816f6b197fe8aebe76d04fa5eb2
```

建议解压到新目录，不覆盖 dev1 代码：

```bash
export DEV2_CODE=/seu_share/home/huangkai/220243809/RAG/SAG/offline-v8-dev2-af66519
rm -rf "$DEV2_CODE"
mkdir -p "$DEV2_CODE"
tar -xzf /上传位置/sag-qwen3-4b-semantic-extraction-af66519.tar.gz -C "$DEV2_CODE"
cd "$DEV2_CODE"
```

检查：

```bash
test -f configs/sag_semantic_extraction_qwen3_4b_v8_dev2.json
test -f scripts/extract_semantics_v8_dev2.sh
```

不要优先使用 PowerShell 生成的 ZIP 在 Linux `unzip`；该 ZIP 可能含反斜杠条目，Linux `unzip` 会警告或失败。`tar.gz` 已验证使用 POSIX 路径。

### 5.2 运行

```bash
conda activate sag-vllm

export INPUT_JSONL=/seu_share/home/huangkai/220243809/RAG/SAG_private/v8-smoke/eval.pilot.inference.private.jsonl
export IDENTITY_MANIFEST=/seu_share/home/huangkai/220243809/RAG/SAG_private/v8-smoke/eval.v8.development-16.manifest.private.jsonl

# 使用服务器现有模型的真实绝对路径；历史位置通常如下
export MODEL_PATH=/seu_share/home/huangkai/220243809/RAG/SAG/SAG/models/Qwen3-4B

export BACKEND=vllm
export RUN_DIR=/seu_share/home/huangkai/220243809/RAG/SAG_private/v8-smoke/run-dev2-001

unset RESUME
unset RETRY_REJECTED

test -f "$INPUT_JSONL"
test -f "$IDENTITY_MANIFEST"
test -d "$MODEL_PATH"
test ! -e "$RUN_DIR"

bash scripts/extract_semantics_v8_dev2.sh
```

入口自动固定：

```text
VLLM_USE_V1=0
VLLM_ATTENTION_BACKEND=XFORMERS
VLLM_ENABLE_PREFIX_CACHING=0
VLLM_ENABLE_CHUNKED_PREFILL=0
VLLM_ENFORCE_EAGER=0
SEMANTIC_LLM_DTYPE=float16
```

只有在 prefix/chunked 已关闭后仍复现同一个 LLVM slice layout 错误，才试 `VLLM_ENFORCE_EAGER=1`。

### 5.3 运行后只回传安全聚合

```bash
for f in \
  "$RUN_DIR/check.safe.json" \
  "$RUN_DIR/diagnostics-summary.safe.json" \
  "$RUN_DIR/quality.safe.json"
do
  printf '\n===== %s =====\n' "$(basename "$f")"
  cat "$f"
done
```

可以回传：

- `check.safe.json`
- `run.safe.json`
- `quality.safe.json`
- `diagnostics-summary.safe.json`
- 文件 SHA-256

绝对不要粘贴：

```text
semantic.private.jsonl
rejects.private.jsonl
candidates.private.jsonl
decisions.private.jsonl
diagnostics.safe.jsonl
```

也不要 `head/cat/less` 私有 JSONL。

### 5.4 dev2 首轮判读顺序

先看结构目标：

- records 是否明显高于 5；
- rejects 是否明显低于 11；
- repairs 是否明显低于 11；
- `malformed_issue_member` 是否显著下降；
- `empty_issue` 是否显著下降；
- `string_issue_member_candidate` 数量；
- `recovered_issue_source`；
- `recovered_issue_surface_grounding`；
- objects/problem_behaviors/request_actions/question_focus coverage。

然后才看新风险：

- 字符串恢复是否过度；
- problem/question/request 角色是否错；
- issue 是否过拆或漏拆；
- location 是否挂错 issue；
- request action 是否进入 problem behavior；
- object/behavior/action attachment 是否制造假共边。

coverage、warnings、accepted rate 都不能冒充 precision/recall。

---

## 6. 已完成的基础设施和历史结果

### 输入与隐私

- multiview input v2；
- PII redaction v2；
- 四个 clean fields：title/content/goal/address；
- 稳定 `doc_id/content_hash`；
- 原 TSV 前100k本地导出98,890条有效记录；
- safety findings 0；
- schema/redaction/hash/identity/quality errors 0；
- inference packet 精确绑定 manifest；
- 新模型入口不消费原 TSV。

### 推理与审计

- Transformers batching、SDPA、dynamic KV；
- vLLM offline continuous batching、paged KV；
- primary/repair 独立 token；
- append/fsync checkpoint；
- privacy-safe diagnostics；
- candidate ledger 先于 semantic checkpoint；
- decision ledger；
- replay 当前 validator；
- exact identity manifest；
- safe checker；
- 每 issue 独立 SAG event；
- DuckDB 保留同 doc 多 issue event；
- metadata 可复制到 issue，正文 regex 不复制到所有 issue。

### 历史 GPU 结果

v7 challenge 32：

- 32 records；
- 0 rejects；
- 3 repairs；
- 0 truncation；
- 约 `278 output tokens/s`；
- GPU peak allocated/reserved `26.712/28.541 GB`。

这只是偏置 challenge set 的结构稳定性，不是生产语义准确率。

v8_dev1 development 16：见第0节。它是失败但有效的结构基线。

### 标注和评测

- 原正式 A/B 标注包仍是 0/48 completed；没有双人人工 adjudicated gold；
- 已完成48条 AI-assisted silver：46条直接接受AI候选、2条人工修改；
- designation：`human_confirmed_ai_assisted_silver`；
- `human_gold_claim_allowed=false`；
- AI subagent 不是人工标注员；
- 旧 issue 定义曾得到146 issue，但受旧 problem/request/history 过拆规则影响，不能作为新 schema 的最终 gold；
- relevance/query 流程已暂停；21个 grade conflict 未完成 adjudication；
- 不得基于旧 relevance、synthetic Oracle 或旧146 issue 宣称正式检索收益。

schema 稳定后，才决定是否按新 issue 定义重组48条 silver，并只人工审核 issue 数量和 attachment；正式论文级结论仍需真正独立双标和第三方仲裁。

---

## 7. 绝对不能再踩的坑

### 数据和隐私

1. 不要提交或聊天粘贴原 TSV、正文、semantic、reject、ledger、DuckDB、ZIP/TAR、模型权重、人工 evidence 或 `*.bak`。
2. 不要把 `SAG_private` 放进 Git。
3. diagnostics 不能包含 `doc_id`、正文、Prompt、evidence、原始响应或可能带正文的异常消息。
4. 不要用 `cat/head/less` 打印私有 JSONL；只看 `.safe.json` 聚合和 SHA-256。
5. current-v1 PII 规则已失效且删除；不得恢复。只用 `sag_pii_redaction_v2`。
6. 原 TSV 只读、本地脱敏；服务器模型只读脱敏输入。

### 实验设计

7. 不要换 seed 重抽 development。固定 seed 是 `sag-v8-split-v1`。
8. exact identity 模式不能用 `LIMIT=16` 替代 manifest。
9. Prompt/schema/decoder/backend A/B 必须使用全新输出目录；不要 resume 或复用目录。
10. checkpoint identity 是 `(doc_id, content_hash, prompt_version, model_id)`；backend 不参与 identity。
11. 不要在同一16条 development 上无限修 Prompt；dev2 后只按聚合错误类别迭代。达到 RC 后必须首次打开冻结 holdout。
12. 不要把 coverage、warning 数、validator 通过率、accepted rate、synthetic Oracle 或AI silver当作 precision/recall。
13. 不要在新 issue schema 稳定前恢复旧 relevance/Oracle 正式结论。
14. 不要先跑995/100k；顺序是 development → 冻结RC → holdout → 48/100扩展 → 995 → 100k。

### Prompt、parser 和语义

15. JSON Schema/guided decoding只约束结构，不解决 issue 边界、attachment、事实或 evidence。
16. dev1 失败不是 token 不够；不要盲目加 token。
17. 字符串 member 只能逐字 grounding 后恢复；不能凭模型语义猜 field/evidence。
18. 无可靠 evidence 的可选候选应删除，不应导致整单 repair/reject。
19. 正常服务动作不是 problem behavior；“希望维修”是 request，“未维修”才是问题事实。
20. question 用 `question_focus/issue_predicate`，request 用 `request_action`；两者默认不进入 expansion frontier。
21. 不要让模型输出 canonical/confidence/issue_id/metadata；surface/evidence 是事实权威。
22. event text 必须是 issue 局部文本，不能无条件复用混合多个问题的整单 summary。
23. emotion/satisfaction/urgency 高精度优先；无直接 evidence 时用空/unknown/normal。
24. 历史答复通常是上下文，不应自动建独立 issue/event。

### GPU / vLLM / 服务器

25. V100 用 FP16，不要 BF16 或 FP32。
26. 固定 vLLM V0 + XFormers + no-prefix + no-chunked-prefill。
27. 只有关闭 prefix/chunked 后仍出现 `LLVM ERROR: Failed to compute parent layout for slice layout.` 才试 eager。
28. 不要终止其他用户GPU进程；`Operation not permitted` 是权限边界。
29. OOM 先确认其他用户占用；不要盲目重装CUDA、删环境或改依赖。
30. 当前服务器不能访问外网；不要要求其访问 GitHub/Hugging Face/PyPI。
31. Linux 离线部署优先用已验证 tar.gz；PowerShell ZIP 的反斜杠条目曾导致 Linux `unzip` 失败。
32. NCCL 的 `destroy_process_group()` 退出警告在 dev1 中没有破坏产物，不是11个reject的原因。

### Git 和工作区

33. 只在功能工作树工作：`G:\RAG\SAG\.worktrees\qwen4b-semantic-extraction`。
34. 不要混入根工作区无关修改：`.gitattributes`、`.gitignore`、`LICENSE`、fixture TSV、`.superpowers/`。
35. 仓库脚本本地运行用 `PYTHONPATH=src`。
36. Prompt/语义变化升级 `prompt_version`；纯 validator/projection/decoder instrumentation 用各自版本。
37. 不要破坏 v7 或 v8_dev1 冻结路径；dev2 必须独立版本化。

---

## 8. 新会话的验证命令

```bash
cd /g/RAG/SAG/.worktrees/qwen4b-semantic-extraction
export PYTHONPATH=src

git status --short --branch
git rev-parse HEAD
git log -3 --oneline --decorate

PYTHONPATH=src python -m unittest discover -s tests
python -m compileall -q src scripts tests
for f in scripts/*.sh; do bash -n "$f" || exit 1; done
git diff --check
```

当前期望：

```text
212 tests OK
af665198adfa3d12bfc6daedc3f0244d3d8a84f0 是 HEAD 的祖先
```

验证实现基线存在：

```bash
git merge-base --is-ancestor af665198adfa3d12bfc6daedc3f0244d3d8a84f0 HEAD
```

冻结检查：

```bash
git diff --exit-code -- \
  src/ragflow_style_pipeline/sag_semantic_prompt.py \
  src/ragflow_style_pipeline/sag_semantic_schema.py \
  src/ragflow_style_pipeline/sag_semantic_validation.py \
  configs/sag_semantic_extraction_qwen3_4b.json \
  tests/test_sag_semantic_prompt.py \
  tests/test_sag_semantic_schema.py \
  tests/test_sag_semantic_validation.py
```

提交前检查私有产物：

```bash
find . -type f \
  \( -name '*.private.*' -o -name '*.duckdb' -o -name '*.bak' -o -name '*.zip' -o -name '*.tar.gz' -o -name '*.tsv' \) \
  -not -path './.git/*' \
  -not -path './tests/fixtures/*' \
  -not -path './.superpowers/*' \
  -print
```

---

## 9. 接下来的决策树

### 如果 dev2 仍大量 `malformed_issue_member`

先确认服务器确实运行：

```text
prompt_version=sag_semantic_v8_dev2
validator_version=sag_semantic_issue_validator_v2
```

若版本正确但仍大量字符串输出：分析 safe warnings，考虑 guided JSON A/B；不要直接扩大数据或加token。

### 如果 dev2 结构成功率显著提高

不要立刻跑 holdout。先在同16条 development 上进行私有人工错误分类，至少看：

- issue 过拆/漏拆；
- object/behavior/action/question 角色；
- attachment；
- location 归属；
- evidence；
- discourse；
- 字符串确定性恢复是否误接。

如果只剩少量、明确、可泛化的错误，可做 `v8_dev3`；否则冻结 RC。

### 如果准备冻结 RC

- 将 Prompt/version 升为 RC；
- 不再看 development 个案改规则；
- 首次运行32条 holdout；
- 只在 holdout 做一次客观评估，不反复调参；
- holdout稳定后扩大48–100，再995，最后100k。

### schema 稳定后

- 按新 issue 定义整理评测 attachment；
- 明确 silver 与真正 human gold；
- 恢复 flat vs issue-aware Oracle 和 relevance；
- 评估 Recall@K、Precision@K、nDCG、MRR、错误扩展率和 hub inflation；
- 不复用旧过拆 issue 的正式结论。

---

## 10. 完成标准

最终不能只看“运行成功”。至少依次满足：

1. dev Prompt 在冻结 development 上结构稳定；
2. 私有人工错误分类确认 issue 边界和 attachment 可用；
3. RC 在独立 holdout 上稳定；
4. 48–100扩展集覆盖生产和challenge；
5. 真正独立双标/仲裁或明确限定为silver；
6. grounded mention、attachment、hyperedge purity 有可复核指标；
7. SAG检索指标和错误扩展率通过；
8. 995用于长尾和性能审计；
9. 最后才跑100k并投影DuckDB。

当前离完成最近的一步只有一个：**部署 `af66519`，在原16条 development 上运行全新 `v8_dev2`，回传三份 safe 聚合报告。**
