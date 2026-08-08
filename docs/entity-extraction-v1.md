# Entity extraction v1：数据、模型与恢复契约

## 1. 设计目的

v1 只解决一个问题：从原始 12345 TSV 生成可审计的脱敏文档和逐字 grounded issue 实体。
它不继承旧版复杂 discourse、canonical、gold、Oracle、数据库或检索代码，也不声称结构成功
等于语义准确或检索收益。

最小 issue 的定义是：**只有把两组实体放在同一组会产生原文不存在的关系时才拆分**。
problem、question、request 是一个现实业务关注点内的不同角色，不能仅因话语角色不同而拆分。

## 2. 原始 TSV

唯一入口是 `data/t_order_master.tsv`，按 UTF-8-SIG、真实 tab 分隔、单物理行记录读取。
header 必须存在、列名非空且唯一，并包含：

- identity：`id`、`order_id`；
- 四个模型语义源：`title`、`case_content`、`case_goal`、`address_detail`；
- metadata：`service_object_type`、三级区域、三级业务分类、`order_source`、
  `order_type`、`order_status`、`call_time`。

每条坏行不会终止全量，但只以以下聚合码进入 `prepare.safe.json`：

- `bad_field_count`
- `missing_source_id`
- `duplicate_source_id`
- `missing_semantic_text`
- `suspected_field_shift`
- `polluted_structured_field`
- `pii_residual`

源 ID 缺失必须拒绝，不能用正文 hash 冒充稳定业务身份。`doc_id` 由命名空间和首个非空的
`id/order_id` 单向 SHA-256 派生，源 ID 不写入任何产物。

`content_hash` 只绑定四个完整脱敏 clean fields。metadata 变化不会触发实体重跑；clean fields
变化会改变 hash。

## 3. PII 与日志边界

四个 clean fields 和所有输出 metadata 都先经过 Python 确定性脱敏，再进入 document、
`rag_text` 或 Prompt。覆盖手机号、身份证/长编号、邮箱、显式姓名、联系人姓名、带标签座机、
QQ、微信等已知模式。脱敏后再做同规则残留扫描；仍命中则拒绝该行。

safe 文件和终端错误不得含：

- `doc_id` 或源 ID；
- 正文、surface、evidence 或 mentions；
- Prompt、模型原始响应；
- 私有输入/输出路径。

`documents.private.jsonl`、`entities.private.jsonl`、`rejects.private.jsonl`、checkpoint、contract
和 links 都是私有文件，不能离开服务器。

## 4. 文档层

`documents.private.jsonl` 是 prepare 原子发布的运行输入快照。每行包含：

- schema/pipeline/redaction version；
- `doc_id` 与 `content_hash`；
- 四个完整脱敏 clean fields；
- `rag_text`：完整脱敏 clean fields 加可靠 metadata；
- metadata：诉求类型、区域、业务分类、来源、工单类型/状态、时间与月份。

metadata 与正文实体分离：例如 `service_object_type=咨询` 只是过滤字段，不冒充正文 question
或 intent。模型视图只使用四个 clean fields，共享 2200 Unicode 字符预算。

## 5. 模型层

生产后端只有 vLLM，无 Transformers fallback。固定：

```text
vllm==0.8.5
transformers==4.51.3
VLLM_USE_V1=0
VLLM_ATTENTION_BACKEND=XFORMERS
VLLM_ENABLE_PREFIX_CACHING=0
VLLM_ENABLE_CHUNKED_PREFILL=0
VLLM_ENFORCE_EAGER=0
VLLM_LOGGING_LEVEL=WARNING
FP16, TP=1, enable_thinking=false
```

不兼容环境 fail closed。只有在关闭 prefix/chunked 后仍复现
`LLVM ERROR: Failed to compute parent layout for slice layout.`，才由人工改配置评估 eager；v1
默认和正式 contract 不自动切换。

输出 schema 只有 `issues` 和五个字符串数组。解析只允许：

- 一个完整 JSON object；
- 可选单个 Markdown JSON fence；
- JSON 容器末尾 trailing comma。

不允许 Python literal、NaN/Infinity、重复 key、解释性尾文或截断补全。

## 6. Grounding 与 issue identity

每个候选按固定字段顺序寻找全部非重叠 occurrence。offset 是 Python Unicode 字符索引，区间
`[start,end)`，并强制：

```python
clean_fields[field][start:end] == text == evidence
```

未命中、纯空白、纯标点或包含 PII placeholder 的候选局部删除。role 内 surface 稳定去重；
grounding 后全空 issue 删除；完全重复 issue 删除。只有整单没有任何有效 issue 才触发 repair。

`issue_id` 由 `doc_id + 各 role 的去重 surface 集合` 稳定派生，不使用 issue 数组下标。
mention offsets 不参与 ID，因此同一实体集合在文本位置变化后 content hash 会变化，但 issue identity
规则仍清晰可重放。

## 7. 推理持久化与 resume

`run.contract.private.json` 绑定：

- documents identity fingerprint；
- model fingerprint；
- Prompt/schema fingerprint；
- 完整 inference config fingerprint。

模型 fingerprint 会 hash `config/tokenizer/*.index.json/*.py/*.jinja` 等小文件内容，并绑定
权重相对文件名和 size；它明确**不是完整权重 hash**，在安全成本和可重放性之间取折中。

checkpoint 事件和 terminal decision 均 append、flush、fsync：

1. `primary_started` 在 primary 调用前；
2. `needs_repair` 在 primary 失败后；
3. `repair_started` 在 repair 调用前；
4. entity/reject 作为 terminal decision。

resume 拒绝未知 identity、重复 terminal、content mismatch、model/prompt/config/contract mismatch。
Linux advisory lock 禁止两个 extract 进程同时追加同一 RUN_DIR。已经开始但未 terminal 的 primary
直接进入唯一 repair；已经开始但中断的 repair 以私有 reject 结束，从而保持“primary 一次、
repair 最多一次”。

## 8. 权威实体与投影

`entities.private.jsonl` 是权威结果：每单包含 grounded issues、五类成员、全部 mentions、
grounding stats、attempt diagnostics 和 contract。`rejects.private.jsonl` 只包含私有 identity、
安全错误码和无文本 telemetry，不保存模型响应。

`entity_links.private.jsonl` 每行是一个 issue member link：

- `event_id`/`issue_id` 表示 issue 超边；
- `member_id`、role、surface 与全部 mentions 表示成员；
- 不含 normalized、canonical 或 confidence。

links 总是从 entities 原子重建，不与推理阶段做跨文件事务。

## 9. Checker 与解释限制

checker 重算 document content hash，检查 terminal 集合、contract、attempt 次数、逐字 grounding、
issue ID 和 projection replay，并生成 `run.safe.json`。它证明的是结构与可重放性，不证明：

- 五个语义角色标注正确；
- issue attachment 正确；
- 实体 coverage 足够；
- RAG/SAG Precision@K、Recall@K、nDCG 或 MRR 提升。

因此 smoke 后必须先做服务器内私有人工审计，再扩大数据规模；全量 accepted 率不能替代准确率。
