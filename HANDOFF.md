# SAG / Qwen3-4B 工单语义抽取交接文档

> 面向一个完全没有历史上下文的新会话。
>
> 最后更新：2026-08-08。本文件只记录代码、版本、聚合指标和私有产物路径，不含工单正文、evidence、`doc_id`、原始模型响应或人工标注内容。

## 0. 新会话先做什么

1. 工作目录切到：

   ```bash
   cd /g/RAG/SAG/.worktrees/qwen4b-semantic-extraction
   export PYTHONPATH=src
   ```

2. 先确认状态，不要立刻改代码或连接服务器：

   ```bash
   git status --short --branch
   git fetch origin
   git rev-parse HEAD
   git rev-parse origin/main
   ```

3. 阅读：
   - 本文件；
   - `docs/13-Qwen4B工单级语义抽取.md`；
   - `docs/14-SAG语义评测与Issue标注.md`。

4. 当前最优先工作不是继续调 Prompt，而是完成 **current-v2 的 A/B 独立人工标注、第三方仲裁、Oracle SAG 实验**。

5. 用户已明确要求：**当前不得连接服务器**。在用户重新明确授权前，不得 SSH，不得在服务器检查目录、GPU、进程或文件。也不要在本地下载、加载或运行 Qwen。

---

## 1. 我们在做什么

项目目标是把 12345 工单转成适合 SAG 检索的、可审计的开放域语义结构。模型为 Qwen3-4B，但目标不是简单 NER，而是每张工单一次联合抽取：

- event；
- problem object；
- problem behavior；
- road / intersection / POI；
- intent；
- emotion；
- satisfaction；
- urgency。

核心约束：

- 每张工单只允许 **一次 primary generation**；
- 仅整单结构/语义异常时，最多 **一次 repair**；
- 多个 entity link 是 Python 下游展开粒度，不能让模型按实体重复读取同一工单；
- 工单级 semantic JSONL 是权威中间产物；
- SAG entity/event/discourse 表只是可以重放的确定性投影；
- 模型输入只能是脱敏 multiview JSONL，不能直接消费原始 TSV；
- 保留完整脱敏 chunk，不能用 event summary 取代事实源；
- 开放域识别，不用固定业务词表限定主题；
- 目标质量分三层评估：
  1. grounded semantic extraction；
  2. issue/hyperedge 关系纯度；
  3. SAG 检索的 Precision@K、Recall@K、nDCG、MRR、错误扩展率等。

当前架构决策是：**先用真实双标 gold 和 Oracle flat vs issue-aware SAG 证明 issue schema 有检索价值，再决定是否让 Qwen 输出 issue frame。** 不允许因为 synthetic 示例或主观直觉直接升级生产 schema。

---

## 2. 仓库、分支与功能基线

### 2.1 路径

- 主仓库：`G:\RAG\SAG`
- 当前功能工作树：`G:\RAG\SAG\.worktrees\qwen4b-semantic-extraction`
- 当前本地分支：`feature/qwen4b-semantic-extraction`
- Git 远程：`https://github.com/Serelity/SAG.git`
- 当前功能代码基线：`24082bf4d42ad3b7dbf73f22e97023138003c4c4`
  - `fix: harden private input redaction`
- 上次核对时，该提交已推送到 `origin/main`，工作树在创建本交接文档前是干净的。

注意：本地分支名仍是 feature，但历史工作一直用 `git push origin HEAD:main` 推送。新会话必须先 fetch/比较，不要假设本地 branch tracking 状态等于远程状态。

### 2.2 最近关键提交

- `24082bf`：扩展 PII 脱敏、安全 gate 和 redaction version；
- `ebfbaad`：challenge 抽样覆盖地址语义；
- `333d8c4`：精确最小 Qwen inference packet；
- `66c0dba`：完整四字段脱敏 multiview 输入；
- `effe0af`：本地私有标注工作台；
- `6c41622`：隔离 A/B annotation round；
- `dda6534`：模型运行精确绑定 `(doc_id, content_hash)`；
- `ee41339`：candidate/decision ledger 对 gold 审计；
- `772baea`：显式第三方仲裁和 final gold merge；
- `a0018b0`：Oracle SAG 检索评估；
- `022eb81`：本地 semantic annotation QA；
- `7083a49`：冻结的 semantic v7 行为基线；
- `2eb61fc`：semantic v6；
- `9950c78`：V100 关闭会触发 LLVM 错误的 prefix kernel 路径；
- `321172f`：vLLM backend。

### 2.3 当前版本契约

见 `src/ragflow_style_pipeline/sag_semantic_versions.py`：

- Prompt：`sag_semantic_v7`
- multiview：`sag_multiview_input_v2`
- PII redaction：`sag_pii_redaction_v2`
- inference packet：`sag_semantic_inference_packet_v1`
- eval manifest：`sag_semantic_eval_manifest_v2`
- gold：`sag_issue_gold_v2`
- annotation round：`sag_issue_annotation_round_v1`
- validator：`sag_semantic_validator_v1`
- projection：`sag_semantic_projection_v1`
- decoder：`unconstrained_json_v1`
- candidate ledger：`sag_semantic_candidate_ledger_v1`
- decision ledger：`sag_semantic_decision_ledger_v1`

当前模型配置在 `configs/sag_semantic_extraction_qwen3_4b.json`：

- model：`Qwen/Qwen3-4B`
- `enable_thinking=false`
- `temperature=0.0`
- `max_input_chars=2200`
- primary `max_new_tokens=640`
- repair `max_new_tokens=768`
- 最多一次 repair
- Transformers 默认 batch 8
- vLLM max num seqs 64

不要随意更改这些值并把结果与 v7 baseline 混在一起。

---

## 3. 私有数据的唯一当前版本

### 3.1 原始源数据

本地原 TSV：

- Windows：`G:\12345_pro_promax\data\t_order_master.tsv`
- Git Bash/WSL：`/g/12345_pro_promax/data/t_order_master.tsv` 或 `/mnt/g/12345_pro_promax/data/t_order_master.tsv`
- 大小：1,245,065,671 bytes
- 约 129 列

原 TSV 只读使用，不能复制进仓库，不能上传 GitHub，不能直接作为模型输入。

### 3.2 当前权威私有根目录

```text
G:\RAG\SAG_private\semantic-eval\current-v2\
```

当前指针：

```text
G:\RAG\SAG_private\semantic-eval\CURRENT.safe.json
```

该指针必须为：

- `current_version=current-v2`
- `redaction_version=sag_pii_redaction_v2`
- multiview SHA-256：`sha256:73440b4ed12f5400e29a5ac199e2274ad642540dfb4e06a729a04d1195ebecd2`

readiness：

```text
G:\RAG\SAG_private\semantic-eval\current-v2\audit\current-v2.readiness.safe.json
```

### 3.3 current-v2 的安全聚合结果

从原 TSV 前 100,000 行本地导出：

- 有效脱敏记录：98,890
- 字段数异常跳过：739
- 四个语义字段全空跳过：242
- 污染文本跳过：129
- checker errors：0
- safety findings：0

字段非空：

- `title_clean`：0
- `case_content_clean`：98,890
- `case_goal_clean`：98,887
- `address_detail_clean`：6,281（6.3515%）

原始数据前 100k 中只有 3 条非空 title，三条都属于字段数异常行，所以最终 `title_clean=0` 是数据结果，不是当前 exporter 又丢了 title。

扩展 PII 脱敏命中：

- email：29
- contact name：15
- labelled landline：176
- QQ：5
- WeChat：2

扩展安全 scanner 对模型可见四个 clean fields 和 metadata 扫描后，所有支持的残留计数均为 0。

### 3.4 current-v1 已经失效并删除

旧 current-v1 的旧安全扫描只覆盖手机号、身份证和较窄的显式姓名规则，后来发现仍有邮箱、座机、联系人、QQ、微信残留。因此：

- current-v1 **绝对不能再用**；
- current-v1 私有目录已删除；
- current-v2-smoke 已删除；
- v1→v2 有 219 条记录的 `content_hash` 改变；
- 98,890 个 `doc_id` 全部保持稳定；
- 迁移报告：`current-v2/audit/redaction-v1-to-v2.migration.safe.json`。

`legacy-proxy-v1/v2` 只是旧代理数据画像，不是 current 数据，不得拿来做最终 gold、最终 manifest 或模型质量结论。旧 proxy-v2 A/B 文件只是 96 条 `in_progress` 空模板，没有正式人工 gold。

---

## 4. current-v2 已生成的产物

### 4.1 输入与安全报告

```text
current-v2/input/t_order_master.100k.multiview.private.jsonl
current-v2/input/t_order_master.100k.quality.safe.json
current-v2/input/t_order_master.100k.safety.safe.json
current-v2/input/t_order_master.100k.check.safe.json
```

multiview JSONL 约 234.66 MB，包含脱敏正文和稳定 identity，仍然是私有数据，不能提交或粘贴。

### 4.2 画像

```text
current-v2/audit/input.profile.safe.json
```

当前 payload 重复画像：

- 当前完整 payload：重复记录率 3.47%；
- 去 time metadata：22.14%；
- semantic metadata only：23.48%；
- clean fields only：24.94%。

这说明未来可能有 inference cache 价值，但 **尚未实现**。不能直接按“去时间 payload”缓存，因为时间 metadata 可能影响语义，且缓存绝不能合并工单事件身份。即使共享模型结果，每张工单仍必须生成独立 order event 和自己的 metadata。

### 4.3 48 条 pilot manifest

```text
current-v2/audit/eval.pilot.manifest.private.jsonl
current-v2/audit/eval.pilot.manifest.report.safe.json
```

- 24 production
- 24 challenge
- challenge 中 6 条有 `address_detail_clean`
- manifest schema：`sag_semantic_eval_manifest_v2`
- manifest content SHA-256：`sha256:9555f66687893b6211fb8c88a68c843da5b4d9d8ba26f444e98000db15ffa962`

manifest 是 private identity 文件，不能提交或粘贴。

### 4.4 A/B 标注轮次

Pristine packet：

```text
current-v2/audit/eval.pilot.annotation.private.jsonl
```

A/B 文件：

```text
current-v2/audit/eval.pilot.annotator-a.private.jsonl
current-v2/audit/eval.pilot.annotator-b.private.jsonl
```

聚合状态：

- round ID：`c8492bb4dc73a3fa`
- source packet SHA-256：`sha256:a024f5868d8263352f9c997e99b700fbe1a4ca92a41c5b090dd50451421c19d6`
- A：48/48 `in_progress`
- B：48/48 `in_progress`
- 当前结构错误：0
- 还没有正式人工标注结果

A/B 必须由两位标注者彼此隔离地完成，不能互看文件或结果。

### 4.5 最小 Qwen inference packet

```text
current-v2/model-input/eval.pilot.inference.private.jsonl
current-v2/model-input/eval.pilot.inference.report.safe.json
current-v2/model-input/eval.pilot.inference.check.safe.json
```

- 48 条
- 48,295 bytes
- SHA-256：`sha256:75b902a8e494faa81cbc41bfb3cc5f14397cc2832436ce492d9cfd7ffa49811a`
- checker errors：0

该包仅保留四个 clean fields、必要 metadata 和精确 identity，仍是私有文件。将来只有在用户明确允许服务器/GPU执行时，才通过私有通道传输它和同一 manifest。不要传完整 100k multiview。

### 4.6 当前没有的产物

本地目前没有 current-v2 的：

- Qwen `semantic.private.jsonl`
- `rejects.private.jsonl`
- candidate ledger
- decision ledger
- diagnostics
- model run report
- model quality report
- adjudicated gold
- Oracle DuckDB / query relevance 集

因此 readiness 正确状态是：

- `ready_for_local_annotation=true`
- `ready_for_model_evaluation=false`
- `model_outputs_available=false`

---

## 5. 已完成的工程工作

### 5.1 脱敏 multiview 输入

已实现：

- `title_clean`
- `case_content_clean`
- `case_goal_clean`
- `address_detail_clean`
- 稳定 `doc_id`
- 稳定 `content_hash`
- `input_schema`
- 独立 `redaction_version`
- 删除脱敏 JSONL 中不必要的原始 `metadata.order_id`
- 四字段全空过滤
- clean fields + metadata 全面污染/PII 扫描
- JSONL/quality 临时文件、`fsync`、原子替换
- 输出 bytes/SHA-256 provenance
- 聚合式 checker，不输出正文或 identity
- `--fail-on-findings` 安全 gate

### 5.2 semantic v7 抽取

已实现并冻结：

- 一工单一次 primary、最多一次 repair；
- compact JSON；
- primary/repair 独立 token 上限；
- batching、SDPA、dynamic KV cache；
- vLLM offline continuous batching/paged KV；
- parser 对 trailing comma、字符串控制字符、受限 Python literal 的保守恢复；
- 拒绝 NaN、Infinity、超范围数字、非 JSON 类型和截断对象补全；
- fixed-point sanitation；
- 无可靠 evidence 的可选候选确定性删除，而不是整单 reject；
- 正常服务动作与异常 problem behavior 区分；
- question/request predicate 与 problem behavior 区分；
- request action 单独保存；
- 严格 road/intersection/POI gate；
- direct emotion/satisfaction/urgency evidence；
- 高风险 canonical evidence 检查；
- candidate/decision ledger；
- privacy-safe diagnostics；
- checkpoint/resume 与 run report。

历史 v7 的 32 条 challenge batch 曾达到 32 records、0 rejects、3 repairs、0 truncation；这只是偏置 challenge set 的结构稳定性结果，不是生产语义精度结论。

### 5.3 评测与审计

已实现：

- deterministic production/challenge manifest；
- 精确 identity manifest `(doc_id, content_hash)`；
- private annotation packet；
- 隔离 A/B annotation round；
- gold enum/field/evidence/identity/provenance validator；
- A/B agreement 和 private conflict packet；
- 第三方显式仲裁；
- adjudicated gold merge；
- semantic mention/attachment/hyperedge 指标；
- flat vs issue-aware Oracle 投影；
- Precision@K、Recall@K、nDCG、MRR、seed recall、false seed rate、one-hop precision、错误扩展率、hub inflation；
- candidate ledger replay；
- ledger 对 gold 的 raw/final PRF、正确/错误保留/删除/新增和状态迁移审计。

Synthetic Oracle 测试能检测 flat event 的跨 issue 假 AND seed 和无关地点扩展，但 synthetic 结果不能冒充真实收益。

### 5.4 本地标注工作台

已实现无第三方前端依赖的 loopback-only 工作台：

- 固定 `127.0.0.1`
- 一次性 bootstrap token
- HttpOnly session cookie
- Host/Origin 检查
- CSP / `no-store`
- 请求体上限
- 无访问日志
- 页面不暴露 `doc_id/content_hash/provenance/annotator`
- 浏览器不能覆盖源字段和 provenance
- revision 乐观锁
- validator
- 临时文件、`fsync`、`os.replace`
- 同目录单份 `*.bak`

工作台页面会显示脱敏正文和 evidence，仍然是私有内容；不能截图、转发 bootstrap URL 或远程监听。

### 5.5 回归状态

功能基线 `24082bf` 上：

- 187 项单元测试通过；
- Python compileall 通过；
- Node `--check` 通过；
- 所有 shell `bash -n` 通过；
- `git diff --check` 通过；
- v7 Prompt/schema/validator 冻结检查通过；
- Git 隐私产物检查通过；
- PowerShell package smoke 通过；
- 包内 114 个文件，禁止的数据扩展名/private 文件为 0。

---

## 6. 当前卡在哪里

### 6.1 首要阻塞：没有真实 adjudicated gold

current-v2 的 A/B 文件都只是 `in_progress` 空轮次。没有双人独立标注和第三方仲裁，因此现在不能可靠回答：

- object / behavior 抽取得准不准；
- location 挂到了正确 issue 吗；
- flat event 是否产生不可接受的 false co-membership；
- validator 删除了正确候选还是错误候选；
- issue-aware projection 是否真的改善 SAG 检索。

这不是代码 bug，而是必须由人工完成的评测数据工作。

### 6.2 没有真实 query relevance 集

Oracle 检索需要 20–30 个真实 12345 业务查询及完整或可证明充分的 relevance 标注。当前尚未建立。没有 relevance，不能用 Precision@K/nDCG/错误扩展率作真实结论。

### 6.3 模型实验暂时阻塞

用户要求当前不连接服务器，本地也不加载 Qwen。因此以下工作暂停：

- current-v2 48 条 Qwen v7 运行；
- guided JSON / XGrammar A/B；
- vLLM 性能 A/B；
- 995 和 100k 推理。

服务器只是将来必须加载 Qwen/GPU 的执行端；parsing/validation/checkpoint 是同一推理事务的一部分。其余导出、画像、标注、replay、审计、Oracle、DuckDB、评估均应在本地运行。

### 6.4 issue-aware schema 尚未获证

现在有工具和 synthetic 测试，但没有真实 gold Oracle 证据。因此不能直接把 Prompt 升级成 issue schema，更不能宣称 issue-aware 一定优于 flat。

---

## 7. 下一步计划（严格按顺序）

## 7.1 立即可做：本地 A/B 双标

A 标注者启动：

```bash
cd /g/RAG/SAG/.worktrees/qwen4b-semantic-extraction
export PYTHONPATH=src
PRIVATE_ROOT=/g/RAG/SAG_private/semantic-eval/current-v2

python scripts/run_semantic_annotation_workbench.py \
  --input "$PRIVATE_ROOT/audit/eval.pilot.annotator-a.private.jsonl" \
  --annotator annotator-a
```

B 标注者在隔离环境/不同时间启动：

```bash
python scripts/run_semantic_annotation_workbench.py \
  --input "$PRIVATE_ROOT/audit/eval.pilot.annotator-b.private.jsonl" \
  --annotator annotator-b
```

要求：

- A/B 不能查看对方文件；
- 不要直接修改 provenance/identity/source fields；
- evidence 必须从当前脱敏原文中选择；
- 按 issue 分组 object、predicate/action、location；
- history 与 current 要分清；
- registered `service_object_type` 不是正文显式 intent；
- emotion/satisfaction/urgency 高精度优先，允许稀疏；
- 标注指南有歧义时先记录并修订指南，不要强行猜。

每位完成后验证：

```bash
python scripts/validate_semantic_gold.py \
  --input "$PRIVATE_ROOT/audit/eval.pilot.annotator-a.private.jsonl" \
  --output "$PRIVATE_ROOT/audit/eval.pilot.annotator-a.validation.safe.json" \
  --require-complete \
  --annotator annotator-a

python scripts/validate_semantic_gold.py \
  --input "$PRIVATE_ROOT/audit/eval.pilot.annotator-b.private.jsonl" \
  --output "$PRIVATE_ROOT/audit/eval.pilot.annotator-b.validation.safe.json" \
  --require-complete \
  --annotator annotator-b
```

## 7.2 比较 A/B 并第三方仲裁

```bash
python scripts/compare_semantic_annotations.py \
  --left "$PRIVATE_ROOT/audit/eval.pilot.annotator-a.private.jsonl" \
  --right "$PRIVATE_ROOT/audit/eval.pilot.annotator-b.private.jsonl" \
  --left-annotator annotator-a \
  --right-annotator annotator-b \
  --output "$PRIVATE_ROOT/audit/eval.pilot.agreement.safe.json" \
  --conflicts "$PRIVATE_ROOT/audit/eval.pilot.conflicts.private.jsonl"
```

第三方 referee 只能在 conflict packet 的 `adjudication` 中填写完整最终结构，不能改 `left/right/clean_fields/source_provenance`。不得用多数投票、自动 union 或自动拼接 issue/evidence。

合并：

```bash
python scripts/merge_adjudicated_gold.py \
  --left "$PRIVATE_ROOT/audit/eval.pilot.annotator-a.private.jsonl" \
  --right "$PRIVATE_ROOT/audit/eval.pilot.annotator-b.private.jsonl" \
  --conflicts "$PRIVATE_ROOT/audit/eval.pilot.conflicts.private.jsonl" \
  --left-annotator annotator-a \
  --right-annotator annotator-b \
  --adjudicator referee \
  --output "$PRIVATE_ROOT/audit/eval.pilot.gold.private.jsonl" \
  --report "$PRIVATE_ROOT/audit/eval.pilot.adjudication.safe.json"
```

随后再次严格验证 final gold。

## 7.3 构建真实业务 query relevance

- 设计 20–30 个真实业务检索查询；
- 不能为了证明 issue-aware 而刻意构造无人使用的词组；
- relevance grade 为 1–3；
- Oracle 语料只有 48 条，因此应审阅全集或建立可证明充分的 pooling；
- 未列出的工单会被当作不相关，漏标会虚增错误扩展率；
- query/relevance/traces 都是 private。

格式和完整说明见 `docs/14-SAG语义评测与Issue标注.md` 第 7 节。

## 7.4 Oracle flat vs issue-aware

有 final gold 后：

```bash
python scripts/build_oracle_sag.py \
  --gold "$PRIVATE_ROOT/audit/eval.pilot.gold.private.jsonl" \
  --mode flat \
  --db "$PRIVATE_ROOT/audit/oracle-flat.duckdb" \
  --report "$PRIVATE_ROOT/audit/oracle-flat.build.safe.json"

python scripts/build_oracle_sag.py \
  --gold "$PRIVATE_ROOT/audit/eval.pilot.gold.private.jsonl" \
  --mode issue-aware \
  --db "$PRIVATE_ROOT/audit/oracle-issue.duckdb" \
  --report "$PRIVATE_ROOT/audit/oracle-issue.build.safe.json"

python scripts/evaluate_oracle_sag.py \
  --flat-db "$PRIVATE_ROOT/audit/oracle-flat.duckdb" \
  --issue-db "$PRIVATE_ROOT/audit/oracle-issue.duckdb" \
  --queries "$PRIVATE_ROOT/audit/oracle.queries.private.jsonl" \
  --output "$PRIVATE_ROOT/audit/oracle.retrieval.safe.json" \
  --traces "$PRIVATE_ROOT/audit/oracle.retrieval.traces.private.jsonl" \
  --cutoffs 5,10
```

只有真实 Oracle 证明 issue-aware 改善检索、降低错误共边/错误扩展且不明显损失相关结果，才进入模型 issue schema 设计。

## 7.5 用户重新授权 GPU 后再跑 Qwen

只传：

- `eval.pilot.inference.private.jsonl`
- 同一 `eval.pilot.manifest.private.jsonl`
- 代码/配置

不要传完整 100k 输入。使用全新输出目录，不设置 `RESUME=1`。必须启用 candidate/decision ledger，以便模型结果回本地后做 validator replay 和 gold 审计。

模型运行完成后先执行 `scripts/check_semantic_run.py`，再通过私有通道取回：

- semantic
- rejects
- candidate ledger
- decision ledger
- diagnostics
- run report
- quality report

聊天中只汇报 safe checker/profile/report/diagnostics summary，不粘贴上述私有 JSONL 内容。

## 7.6 后续正交实验顺序

1. 冻结 v7 baseline；
2. current-v2 gold 与 Oracle；
3. 简单 guided JSON/XGrammar；
4. flat schema vs 轻量 issue frame；
5. 空间 soft-candidate union；
6. Prompt 压缩；
7. 如果证明安全，再实现 inference payload cache；
8. gold 指标通过后跑 995；
9. 995 通过后跑 100k、SAG projection 和 DuckDB。

每次只改变一个因素，使用冻结的同一 manifest 和全新输出目录。

---

## 8. 绝对不要再踩的坑

### 8.1 隐私与数据

1. **不要使用 current-v1。** 它的旧 PII 审计漏掉邮箱、座机、联系人、QQ、微信，且已删除。
2. **不要把 legacy proxy 当 current 数据。** proxy 只用于早期画像。
3. 不提交或粘贴：原 TSV、正文 JSONL、semantic、rejects、links、DuckDB、日志、模型输出、人工标注、candidate/decision ledger、ZIP、权重、`*.bak`。
4. diagnostics 不能包含 `doc_id`、正文、Prompt、evidence、原始响应或可能带正文的 exception message。
5. safe report 只能有聚合信息；不要输出 type1/type2/type3 的原始类别值，因为任意值也可能泄露文本。
6. 原始 TSV 只读；下游模型只消费脱敏 multiview/inference packet。
7. 最小 inference packet 仍含脱敏正文和 identity，仍然是 private。
8. 工作台只允许 `127.0.0.1`；不要添加远程绑定参数，不要转发 bootstrap URL，不要截图。
9. 工作台产生的 `*.bak` 也是私有数据，不能提交。

### 8.2 标注与 gold

1. A/B 必须隔离，不能互看。
2. 只能从 pristine `unlabeled` packet 生成 round；不要手工复制或重建 provenance。
3. compare 要求 A/B 的 round provenance 完全一致，不要混用不同 round 或 source packet。
4. 第三方仲裁者必须不同于 A/B；不得自动多数投票、自动 union evidence 或自动拼接 issue。
5. 不要用结构 validator 通过、coverage、warning 数量或 `accepted_with_warnings` 比例冒充 precision/recall。
6. 48 条 gold 是 pilot，不是 100k 生产分布的完整代表。

### 8.3 模型与 Prompt

1. 每工单一次 primary，整单异常最多一次 repair；不要按 entity 重复调用模型。
2. 无 evidence 的可选候选应确定性删除/回退，不应引发整单 repair/reject。
3. Prompt 或语义行为变化必须升级 `prompt_version`；纯 instrumentation 用 validator/projection/decoder version，不要滥升 Prompt。
4. 新 Prompt、guided decoding 或 backend A/B 必须用全新目录；不要拿旧 checkpoint 混跑。
5. checkpoint identity 是 `(doc_id, content_hash, prompt_version, model_id)`；backend 不属于 identity。
6. manifest 必须按 `(doc_id, content_hash)` 精确选择，不能只用 doc_id，也不能用 `LIMIT` 代替 manifest。
7. manifest 模式会完整扫描输入；不要先截断输入导致目标缺失。
8. guided JSON 只解决结构合法性，不解决事实、evidence、canonical 和关系准确率。
9. `enable_thinking=false`，不要为了“更聪明”打开 thinking，当前优先 JSON 完整性、速度和可控 token。
10. `finish_reason=stop` 的 JSON 错误通常不能靠盲目增加 token 解决。
11. 不要在没有真实异常证据时盲调 CUDA、重装依赖、删 partial/diagnostic 文件。

### 8.4 V100 / vLLM

V100 路径固定：

```bash
VLLM_USE_V1=0
VLLM_ATTENTION_BACKEND=XFORMERS
VLLM_ENABLE_PREFIX_CACHING=0
VLLM_ENABLE_CHUNKED_PREFILL=0
VLLM_ENFORCE_EAGER=0
```

并且：

- 使用 `float16`，不要 `bfloat16` 或 FP32；
- vLLM 独立 Python 3.11 环境；
- 固定 `vllm==0.8.5`、`transformers==4.51.3`；
- 关闭 prefix caching 是因为历史出现 `LLVM ERROR: Failed to compute parent layout for slice layout.`；
- 只有关闭 prefix caching 后仍出现同一 LLVM 错误，才试 `VLLM_ENFORCE_EAGER=1`；
- 共享 GPU OOM 时不要杀其他用户进程；`Operation not permitted` 表示没有权限，应等待独占资源或联系管理员。

当前用户没有授权连接服务器，所以这些只是未来运行约束，不是现在要执行的命令。

### 8.5 语义与 SAG

1. 32 条历史 smoke/challenge 偏向空间、历史答复、多句文本，不代表生产分布。
2. emotion、satisfaction、urgency 天然稀疏，高精度优先；低 coverage 不等于失败。
3. 正常服务动作不是异常 problem behavior。
4. question/request predicate 使用 `issue_predicate`，request action 使用 `request_action`，不能塞进 `problem_behavior` frontier。
5. `service_object_type` 是源系统 registered request type，不等于正文显式 intent 或 interaction mode。
6. discourse 默认作为 event/issue attributes，不作为 SAG expansion frontier，避免“咨询/求助/不满”超级节点。
7. canonical/alias 有图连接风险；surface/evidence 是权威。未经验证的模型 canonical 不要直接建全局概念节点。
8. 地址实体使用模型、规则、metadata 和未来 gazetteer 的软候选并集，允许 NIL/OOV；不要用固定词表封闭开放域。
9. mention F1 高不代表 SAG 正确；必须看 issue attachment、false co-membership、hyperedge purity 和检索错误扩展。
10. synthetic Oracle 只能验证评测器是否能发现问题，不能证明真实 issue-aware 收益。
11. inference cache 不能合并事件身份。不要未经实验就按“去时间 payload”复用模型结果。

### 8.6 本地工程与 Git

1. 本地脚本使用 `PYTHONPATH=src`，否则会报 `ModuleNotFoundError`。
2. Windows PowerShell 5 可能按本地代码页误解无 BOM 脚本中的中文文件名；package 脚本已改用 ASCII pattern，不要改回中文 glob。
3. 不要把根工作区无关修改混进提交。历史上明确不应混入：`.gitattributes`、`.gitignore`、`LICENSE`、`tests/fixtures/t_order_master_sample.tsv`、`.superpowers/`。
4. 一个 cwd/worktree 同时只保留一个 writer；审查和验证尽量只读。
5. 提交前执行完整测试、compile/static/freeze/privacy/package 检查。

---

## 9. 关键源码与脚本

### 输入/脱敏

- `src/ragflow_style_pipeline/document_builder.py`
- `src/ragflow_style_pipeline/export_jsonl.py`
- `src/ragflow_style_pipeline/pii_redactor.py`
- `src/ragflow_style_pipeline/scan_jsonl_safety.py`
- `src/ragflow_style_pipeline/multiview_export_check.py`
- `scripts/check_multiview_export.py`
- `scripts/profile_semantic_input.py`

### Qwen semantic

- `src/ragflow_style_pipeline/sag_semantic_prompt.py`
- `src/ragflow_style_pipeline/sag_semantic_schema.py`
- `src/ragflow_style_pipeline/sag_semantic_validation.py`
- `src/ragflow_style_pipeline/sag_semantic_llm.py`
- `src/ragflow_style_pipeline/sag_semantic_projection.py`
- `scripts/extract_semantics_qwen3_4b.sh`
- `scripts/check_semantic_run.py`
- `scripts/summarize_semantic_diagnostics.py`

### 评测/标注/Oracle

- `src/ragflow_style_pipeline/sag_semantic_audit.py`
- `src/ragflow_style_pipeline/sag_oracle.py`
- `src/ragflow_style_pipeline/sag_annotation_workbench.py`
- `src/ragflow_style_pipeline/sag_annotation_server.py`
- `src/ragflow_style_pipeline/semantic_inference_packet.py`
- `scripts/build_semantic_eval_manifest.py`
- `scripts/build_private_annotation_packet.py`
- `scripts/prepare_semantic_annotation_round.py`
- `scripts/run_semantic_annotation_workbench.py`
- `scripts/validate_semantic_gold.py`
- `scripts/compare_semantic_annotations.py`
- `scripts/merge_adjudicated_gold.py`
- `scripts/evaluate_semantic_gold.py`
- `scripts/build_oracle_sag.py`
- `scripts/evaluate_oracle_sag.py`
- `scripts/replay_semantic_candidates.py`
- `scripts/audit_semantic_ledger.py`

### 文档

- `docs/13-Qwen4B工单级语义抽取.md`
- `docs/14-SAG语义评测与Issue标注.md`

---

## 10. 建议的验证命令

修改代码后至少执行：

```bash
cd /g/RAG/SAG/.worktrees/qwen4b-semantic-extraction
export PYTHONPATH=src

python -m unittest discover -s tests -v
python -m compileall -q src tests scripts
node --check src/ragflow_style_pipeline/sag_annotation_assets/app.js
for f in scripts/*.sh; do bash -n "$f" || exit 1; done
git diff --check
```

若没有明确要改 semantic v7 行为，确认 Prompt/schema/validator 未意外变化：

```bash
git diff --exit-code -- \
  src/ragflow_style_pipeline/sag_semantic_prompt.py \
  src/ragflow_style_pipeline/sag_semantic_schema.py \
  src/ragflow_style_pipeline/sag_semantic_validation.py
```

检查不能进入 Git 的数据扩展名/private 文件：

```bash
if git status --porcelain=v1 | grep -E '(^|/)(data|outputs|models|packages)/|\.(jsonl|duckdb|zip|npy|tsv|csv)$'; then
  echo "private/data artifact detected"
  exit 1
fi
```

不要为了验证而读取或打印私有 JSONL 正文；优先读取 `.safe.json` 聚合报告。

---

## 11. 完成标准

不能因为“运行没报错”就宣布完成。最终需要依次满足：

1. current-v2 A/B 48 条均完成并通过 strict validator；
2. 完成 agreement、冲突审阅和第三方 adjudicated gold；
3. 完成真实业务 query relevance；
4. Oracle flat vs issue-aware 有可复核结果；
5. 用户授权后，冻结 manifest 上完成 Qwen v7 baseline 和 candidate/decision ledger；
6. semantic mention、issue attachment、hyperedge purity 与 SAG 检索三层指标共同通过；
7. 正交实验明确 guided JSON、issue frame、空间候选、Prompt 压缩各自贡献；
8. 995 用于长尾发现和性能验证；
9. 最后才运行 100k、SAG projection 和 DuckDB。

在第 1–4 步完成前，最正确的下一步通常是推进标注和 Oracle，而不是继续堆 validator 规则或修改 Prompt。
