# Qwen3-4B 工单级 SAG 语义抽取设计

日期：2026-08-04

状态：已确认，待实施计划

适用项目：`SAG` 12345 工单检索实验

## 1. 背景与目标

当前实体抽取管线使用本地 Qwen3-4B，一条工单调用模型一次，模型返回多个实体，Python 再把每个实体写成一行 `SagEntityLink`。因此“一个 entity 一行”只是关系存储粒度，并不会使模型按实体重复读取工单。

现有实现的主要问题不是调用次数，而是：

- 输入默认指向原始 TSV，没有统一消费脱敏多视图数据；
- 工单级原始语义结果未保存，难以审计、重放和重新投影；
- Prompt 的类型边界和字段语境不足，Qwen3-4B 容易混淆问题事实、诉求动作、历史答复和当前立场；
- `normalized_value` 缺少稳定的 canonicalization，无法可靠支持跨工单连接；
- 模型自报置信度集中在 `0.9`，缺乏校准价值；
- 缺少意图、情绪、满意度和紧急程度等 discourse 分析；
- 普通 Transformers 静态 batch 存在长短文本 padding、冗长 JSON 生成和可观测性不足等问题。

本设计采用“方案 B”：

> 每条工单进行一次 Qwen3-4B 主推理，同时生成一个语义完整 event、一组开放式 SAG 索引实体和 discourse 属性；保存一条工单级权威中间记录，再由 Python 确定性地展开为多条 SAG event-entity links。仅校验异常的工单允许一次选择性修复调用。

目标包括：

1. 开放式识别任意 12345 主题，不局限于少量固定词表；
2. 保持 SAG 的 chunk-event-entity 结构和后续 SQL/向量检索升级路径；
3. 道路、路口、POI 等空间实体可用于 query-time expansion；
4. discourse 可用于过滤、排序、统计和预警，但不默认充当扩展实体；
5. 输入、结果、验证和运行元数据均可审计；
6. 模型和真实数据不在本地运行，GPU 推理与真实数据验证在服务器完成。

## 2. 范围与非目标

### 2.1 本次范围

- 统一的脱敏多视图输入契约和稳定 `doc_id`；
- 工单级 Qwen3-4B Prompt 与结构化输出；
- event、主题实体、空间实体和 discourse 联合抽取；
- 确定性解析、验证、告警和最多一次选择性修复；
- 工单级语义 JSONL 向 SAG links/discourse 表的确定性投影；
- 运行可观测性、断点续跑和服务器操作脚本；
- 不加载模型的本地单元测试与服务器验收流程。

### 2.2 非目标

- 本次不在本地下载、加载或运行 Qwen 模型；
- 不在本地读取或提交原始真实工单；
- 不在第一阶段实现完整在线问答、LLM rerank 或所有向量索引；
- 不让固定主题词表决定模型可以识别的主题范围；
- 不把 emotion、satisfaction 等低区分度属性作为默认图扩展 frontier；
- 不让模型输出或暴露 chain-of-thought；
- 不承诺仅靠规则安全扫描即可证明所有自由文本隐私均已清除。

## 3. SAG 对齐原则

本设计与 SAG 的映射如下：

| SAG 概念 | 12345 设计 |
|---|---|
| chunk | 一条完整脱敏工单 |
| event | 当前工单的语义完整 `event_summary` |
| entities | 问题对象、问题行为、道路、路口、POI，以及由 metadata 确定性生成的区域等实体 |
| chunk/event → entity | 投影后的 `sag_event_entity_links` |
| event attributes | discourse、时间、状态和结构化工单属性 |
| source evidence | `source_field` + 原文连续 `evidence` |
| entity identity | `canonical` 与实体类型共同确定 |
| query-time expansion | 基于高区分度主题/空间实体的 SQL join；未来可结合实体和 event 向量 |

必须保留完整脱敏 chunk，event summary 不是唯一事实源。event 必须是语义完整的当前事件，而不是关键词拼接。entities 是具有跨工单检索或连接价值的索引概念，不是普通 NER 的所有名词和动词。

`discourse` 属于领域扩展，并作为 event attributes 保存：

- 可用于 SQL 过滤、排序、统计和预警；
- 不默认写成可扩展的普通实体；
- 避免用“不满”等超级节点连接语义无关的工单。

## 4. 总体架构

```text
统一脱敏多视图 JSONL
        ↓
确定性预处理
  ├─ 字段适配与稳定 doc_id
  ├─ metadata 实体直接保留
  ├─ 长文本窗口选择
  └─ 历史答复 / 当前诉求提示
        ↓
Qwen3-4B 每工单一次主推理
  ├─ event summary
  ├─ problem objects / behaviors
  ├─ roads / intersections / POIs
  └─ intent / emotion / satisfaction / urgency
        ↓
结构和语义质量闸门
  ├─ JSON/schema 校验
  ├─ evidence 校验
  ├─ 类型形态与字段语境校验
  ├─ canonical 过度改写检测
  ├─ 模板“谢谢”误判检测
  └─ 历史答复污染检测
        ↓
仅 repair_required 的工单最多修复一次
        ↓
work_order_semantics JSONL（每工单一行）
        ↓ Python 确定性投影
  ├─ SAG event-entity links（每关系一行）
  └─ SAG event discourse（每 event 一行）
        ↓
DuckDB / 后续 SQL + 向量 SAG 检索
```

推理粒度与存储粒度严格分离：

- 模型主推理：一条工单一次；
- 工单级中间结果：一条工单一行；
- SQL 关系投影：一个 event-entity 关系一行；
- 多行 link 由 Python 展开，不引发任何模型重读。

## 5. 统一脱敏输入契约

### 5.1 推荐格式

```json
{
  "schema_version": "2.0",
  "doc_id": "order_<stable_hash>",
  "title_clean": "...",
  "case_content_clean": "...",
  "case_goal_clean": "...",
  "address_detail_clean": "...",
  "metadata": {
    "service_object_type": "求助",
    "area_code_city": "常州市",
    "area_code_area": "武进区",
    "area_code_street": "丁堰街道",
    "type1": "城乡建设",
    "type2": "市容管理",
    "type3": "无照经营游商",
    "call_time": "2024-06-11 20:51:18",
    "call_month": "2024-06"
  },
  "content_hash": "sha256:<hash>"
}
```

### 5.2 约束

- 所有进入模型的自由文本必须已经脱敏；
- `doc_id` 必须由统一算法稳定生成，不能因导出版本改变；
- `content_hash` 用于判断内容是否变化和安全断点续跑；
- 旧 `{doc_id, text, metadata}` JSONL 由明确的适配器解析标记段，不能被当前 reader 静默读成空正文；
- 解析失败或关键字段为空必须显式拒绝，不允许继续调用模型；
- 原始 TSV 兼容入口可以保留给受控服务器环境，但生产抽取脚本默认输入必须改为脱敏多视图 JSONL；
- 实际工单文件、模型输出和抽样文本不得提交 GitHub。

### 5.3 metadata 与正文实体

`area_code_area`、`area_code_street`、时间、工单分类等结构化值由 Python 确定性转换，不消耗 Qwen 输出 token。登记区域和正文提及地点语义不同，link 必须保留来源角色，例如：

```text
source_channel = metadata | llm
location_role = registered | mentioned
```

第一阶段 Qwen 重点抽取具体道路、路口和 POI。后续若需要正文区县/街道冲突分析，可扩展 `mentioned_area` 和 `mentioned_street`，不得覆盖 metadata。

## 6. 工单级语义输出 Schema

每条工单输出一条记录：

```json
{
  "schema_version": "2.0",
  "doc_id": "order_xxx",
  "content_hash": "sha256:...",
  "event": {
    "summary": "市民反映港龙新港城北门存在流动摊贩占道经营，希望部门处理",
    "evidence_fields": ["case_content_clean", "case_goal_clean"]
  },
  "entities": {
    "problem_objects": [
      {
        "surface": "电动车摆摊",
        "canonical": "流动摊贩",
        "source_field": "case_content_clean",
        "evidence": "10几个电动车摆摊"
      }
    ],
    "problem_behaviors": [
      {
        "surface": "占道经营",
        "canonical": "占道经营",
        "source_field": "case_content_clean",
        "evidence": "关于占道经营的问题"
      }
    ],
    "roads": [],
    "intersections": [],
    "pois": [
      {
        "surface": "港龙新港城",
        "canonical": "港龙新港城",
        "source_field": "case_content_clean",
        "evidence": "港龙新港城北门口"
      }
    ]
  },
  "discourse": {
    "intents": [
      {"label": "求助", "evidence": "希望清理一下摆摊"}
    ],
    "emotions": [
      {"label": "不满", "intensity": 1, "evidence": "这里快变成夜市了"}
    ],
    "satisfaction": {
      "label": "unknown",
      "target": "",
      "evidence": ""
    },
    "urgency": {
      "level": "normal",
      "evidence": ""
    }
  },
  "validation": {
    "status": "accepted",
    "warnings": [],
    "repair_attempted": false
  },
  "model_run": {
    "model": "Qwen3-4B",
    "prompt_version": "sag_semantic_v2",
    "backend": "transformers",
    "input_tokens": 0,
    "output_tokens": 0,
    "finish_reason": "stop",
    "latency_ms": 0
  }
}
```

模型只生成 `event`、`entities` 和 `discourse` 的业务字段。`schema_version`、`doc_id`、`content_hash`、`validation` 和 `model_run` 由程序填充或覆盖，防止模型伪造运行元数据。

### 6.1 event

- summary 只表达当前核心事件，建议不超过 80 个汉字；
- 优先当前事实、当前立场和当前诉求；
- 历史答复只在理解当前立场所必需时出现；
- 咨询、建议、表扬等工单不得强行制造问题；
- 不加入原文没有的事实；
- 保留 event 到完整脱敏 chunk 的 `doc_id` 关联。

### 6.2 problem objects

问题、诉求或咨询指向的领域对象，开放式识别，不受固定词表限制。例如路灯、电梯、房屋屋面、培训机构、停车收费服务、劳动报酬、食品、物业服务、体检报告。

禁止输出没有检索价值的泛词，例如“服务对象、相关部门、工作人员、事情、情况、问题”。

### 6.3 problem behaviors

定义为与对象相关的问题行为、问题现象、异常状态或核心关系，例如占道经营、违规收费、拖欠工资、照明故障、房屋渗漏、退款未到账。

必须排除纯诉求动作，例如“要求处理、希望维修、请求清理、建议拆除”。例如“树枝遮挡交通标志，希望修剪”应抽取“行道树 + 遮挡交通标志”，不能把“修剪”当成问题行为。

### 6.4 locations

- `road` 只接受具体命名道路，不能接受“马路边、道路、消防通道、地下室、小区北门口”；
- `intersection` 必须有明确路口表达或可验证的道路组合；
- `poi` 包括具体小区、市场、学校、医院、商场、公园、机构等；
- “港龙新港城北门口”应归为 `poi=港龙新港城`，不能归为 road；
- 空间专名 canonicalization 必须保守，不能猜测原文未提供的全称。

### 6.5 surface、canonical 与 evidence

- `surface`：忠实保留原文实体表达；
- `canonical`：用于跨工单实体身份和 join 的标准概念；
- `evidence`：对应 `source_field` 中可定位的连续原文片段；
- canonical 可以归一近义表达，但不能改变事实或把不同业务问题过度合并；
- 程序必须保留 surface/evidence 用于审计，同时使用 `(entity_type, canonical)` 形成实体身份。

## 7. discourse 设计

### 7.1 intents

允许最多三个标签，限定为：

```text
投诉、举报、求助、咨询、建议、表扬、催办、反馈、其他
```

模型推断意图和 metadata 声明类型分开保存。Python 计算：

```text
declared_intent
inferred_intents
intent_conflict
```

正文不能被 metadata 覆盖，metadata 也不能被模型覆盖。

### 7.2 emotions

限定标签：

```text
愤怒、不满、焦虑、无奈、悲伤、感谢、认可
```

强度：`1=轻度，2=明显，3=强烈`。没有直接证据时输出空数组，不默认推断“平静”。必须区分“被投诉对象态度恶劣”和“诉求人表达愤怒”。

### 7.3 satisfaction

标签限定为：

```text
satisfied、dissatisfied、mixed、unknown
```

非 `unknown` 必须同时提供评价对象 `target` 和直接证据。模板性“请优先处理，谢谢”“感谢转交”不能推断满意。历史答复中的“已处理”不能覆盖当前“不认可、仍未解决”。

### 7.4 urgency

限定为：

```text
normal、high、critical
```

- `normal`：普通诉求或无明确紧急证据；
- `high`：明确催办、长期未解决、影响持续扩大；
- `critical`：当前存在人身安全、火灾、燃气泄漏、坍塌等紧迫风险。

平台模板中的“优先处理”不能单独触发 high。

## 8. Prompt 设计

### 8.1 固定结构

Prompt 分为：

1. 角色与任务：面向 SAG 的工单语义结构化，不是普通关键词抽取；
2. 字段定义：当前事实、诉求目标、地址和 metadata 的语义角色；
3. 正反边界：对象/行为、道路/POI、当前诉求/历史答复、礼貌用语/满意度；
4. 输出约束：固定 JSON schema、数量上限、空数组规则、只输出 JSON；
5. 少量覆盖不同领域的困难 few-shot；
6. 当前脱敏工单 payload。

Prompt 要求模型在内部完成判断，但禁止输出分析过程或 chain-of-thought。

### 8.2 判断顺序

Prompt 指导模型依次判断：

1. 当前工单的核心事件；
2. 历史工单、部门答复和当前诉求的边界；
3. 当前最新事实、立场和要求；
4. 具体问题对象；
5. 问题行为、现象或状态；
6. 哪些动作只是诉求人希望部门采取的动作；
7. 具体道路、路口和 POI；
8. 有直接证据的意图、情绪、满意度和紧急性；
9. 每项 evidence 是否存在；
10. 实体是否具有跨工单检索价值。

### 8.3 数量上限

| 字段 | 上限 |
|---|---:|
| problem_objects | 3 |
| problem_behaviors | 4 |
| roads | 4 |
| intersections | 2 |
| pois | 4 |
| intents | 3 |
| emotions | 2 |

上限不是最低数量；没有可靠证据时必须输出空数组。

### 8.4 few-shot 边界覆盖

Prompt 示例至少覆盖：

1. 路灯不亮：对象=路灯、行为=照明故障、road=和平路，维修不是问题行为；
2. 港龙新港城北门：POI，不是 road；
3. 树枝遮挡交通标志：遮挡是问题，修剪是诉求动作；
4. 培训机构闭店并要求退款且末尾“谢谢”：satisfaction=unknown；
5. 收费员拒绝开票且态度恶劣：问题行为不等于诉求人愤怒；
6. 前次答复“已处理”但当前“不认可、仍被占用”：以当前立场为准并识别 dissatisfaction。

few-shot 必须跨领域，不能全部围绕流动摊贩。

## 9. 长文本与历史答复处理

当前简单截取正文前 `max_text_chars` 可能丢失位于末尾的当前诉求。预处理应按 token 预算生成模型输入：

- 短文本完整保留；
- 长文本保留正文首部、当前立场关键短语附近窗口和尾部；
- 识别“前期反映、原工单、处理结果、部门答复、答复如下”等历史提示；
- 识别“其不认可、现服务对象表示、现再次反映、仍未解决、再次要求、希望部门”等当前提示；
- 不以规则直接断言语义，只向模型提供明确段落/窗口和来源标签；
- 截断策略必须记录 `input_truncated`、原字符/token 数和保留窗口信息。

## 10. 解析与验证质量闸门

### 10.1 不采用模型自报 confidence

删除模型生成的实体 `confidence`。首轮结果高度集中于 `0.9`，没有有效区分度。最终质量状态由程序根据可验证信号产生。

### 10.2 验证项目

- JSON 是否完整、是否因 token 限制截断；
- schema 类型、枚举和数量是否正确；
- evidence 是否为对应 source field 的连续原文片段；
- surface 是否与 evidence 一致；
- canonical 是否为空、泛化或过度改写；
- road/intersection/POI 是否存在明显形态冲突；
- problem behavior 是否只是 case goal 中的诉求动作；
- satisfaction 是否有明确 target/evidence；
- 模板“谢谢、优先处理”是否被误判；
- 历史答复是否覆盖当前立场；
- 是否重复输出相同 `(group, canonical, evidence)`；
- 是否超过实体数量上限。

验证状态限定为：

```text
accepted
accepted_with_warnings
repair_required
rejected
```

所有告警使用稳定机器码，不能只写自由文本。

## 11. 选择性修复

只有 `repair_required` 才触发第二次调用，每条工单最多一次。Repair Prompt 包含：

- 原工单必要脱敏片段；
- 原始模型输出；
- 验证器发现的具体错误码和字段路径；
- 只修复指定字段、其他字段保持不变的要求。

适用情况包括：

- JSON 解析失败或截断；
- evidence 不存在；
- 道路与 POI 明显冲突；
- canonical 与证据无关；
- 非 unknown satisfaction 缺乏直接证据；
- 高概率历史答复污染；
- 数量或枚举违反 schema。

修复仍失败则写入 rejects，保留原响应、修复响应、错误码和运行元数据，不静默丢失。敏感文本只允许存在于服务器受控产物中，不进入日志或 Git。

## 12. 两层存储与 SAG 投影

### 12.1 权威中间产物

```text
outputs/work_order_semantics.qwen3_4b.jsonl
```

一条工单一行，包含 event、entities、discourse、validation 和 model_run。它用于：

- 人工审计和质量抽样；
- Prompt/模型版本比较；
- 失败重放；
- 不重跑模型地重新生成下游表；
- 定位语义错误与投影错误。

### 12.2 SAG 查询投影

Python 确定性生成：

```text
outputs/sag_event_entity_links.qwen3_4b.jsonl
outputs/sag_event_discourse.qwen3_4b.jsonl
```

并可加载到：

```text
sag_chunks / source_orders
sag_events
sag_entities
sag_event_entity_links
sag_event_discourse
```

建议 link 增加：

```text
surface_form
canonical_value
source_field
source_channel
location_role
matched_text/evidence
validation_status
prompt_version
```

实体身份由 `(entity_type, canonical_value)` 确定。`event_text` 使用语义完整 summary；完整脱敏 chunk 仍通过 `doc_id` 可追溯。

### 12.3 扩展角色

默认角色：

- seed/index：`problem_object`、`problem_behavior`；
- spatial frontier：`road`、`intersection`、`poi`；
- metadata/filter：`area`、`street`、time、department、case type；
- event attributes：intent、emotion、satisfaction、urgency。

这些是默认策略而非永久硬编码，查询配置应显式控制允许的 seed 和 frontier 类型。

## 13. 输出产物与可观测性

服务器运行产生：

```text
outputs/work_order_semantics.qwen3_4b.jsonl
outputs/work_order_semantics.rejects.jsonl
outputs/work_order_semantics.run.json
outputs/work_order_semantics.quality.json
outputs/sag_event_entity_links.qwen3_4b.jsonl
outputs/sag_event_discourse.qwen3_4b.jsonl
```

`run.json` 至少记录：

- 模型 ID、模型路径或版本摘要；
- Prompt/schema/config 版本及哈希；
- 输入路径的安全标识、行数和 content hash 摘要；
- backend、dtype、设备、batch 设置；
- processed/accepted/warned/repaired/rejected 数量；
- input/output token 总数和分位数；
- 每 batch 和每工单延迟统计；
- finish reason、截断、解析失败和 OOM 计数；
- 吞吐与总耗时；
- 启动/结束时间和断点信息。

`quality.json` 至少记录：

- 各类实体与 discourse 覆盖率；
- 每工单实体数量分布；
- canonical/surface 比例；
- 各 warning/reject/repair 原因分布；
- metadata 与推断意图冲突率；
- 模板谢谢误判拦截计数；
- 不含真实正文的汇总指标。

原始响应建议保存于服务器受控的诊断文件，至少为失败/修复项保留；日志不得打印完整工单。

## 14. 性能与推理后端

### 14.1 第一阶段：兼容 Transformers

- 按 tokenizer 估算长度分桶，短、中、长文本分别 batching；
- batch 内尽量减少 padding；
- 记录真实 input/output tokens 和生成长度；
- 依据服务器显存和实测自动/手动调整 batch size；
- 避免用固定 `max_new_tokens` 掩盖截断，必须检测生成终止状态；
- 主推理一次，只有异常项修复一次；
- metadata 确定性提取，避免模型重复输出；
- 支持 checkpoint、resume 和只重跑失败/指定 doc_id。

### 14.2 第二阶段：可选 vLLM

在 schema 和质量稳定后，可增加 vLLM backend，利用：

- continuous batching；
- prefix caching；
- paged KV cache；
- 更好的长短请求调度；
- finish reason 和 token 统计。

抽取器通过统一生成接口隔离 backend，业务解析和验证逻辑不能依赖某个推理引擎。vLLM 不是第一版正确性的前置条件。

## 15. 断点续跑与幂等性

服务器长任务必须支持安全续跑：

- checkpoint 使用 `doc_id + content_hash + prompt_version + model_version` 作为处理身份；
- 输出采用追加或分片写入，完成后原子合并；
- 启动时扫描已完成身份，跳过完全一致记录；
- 内容、Prompt 或模型版本变化时不得错误复用旧结果；
- 每批 flush 并写 checkpoint；
- 支持 `--retry-rejected`、`--doc-id-file` 和 `--resume`；
- 重跑投影不需要模型；
- 异常退出不得留下看似完整但实际截断的最终文件。

## 16. 测试与验收

### 16.1 本地允许执行

本地只执行不加载模型、不访问真实工单的测试：

- 输入适配器和稳定 doc_id 测试；
- 旧脱敏 JSONL 解析测试；
- Prompt 结构快照/关键规则测试；
- JSON 解析、schema 和 evidence 校验测试；
- road/POI、诉求动作/问题行为、模板谢谢等规则测试；
- 工单级语义结果到 SAG links/discourse 的投影测试；
- checkpoint/resume 幂等性测试；
- 使用人工构造、完全脱敏 fixture 的单元测试；
- lint、静态检查和普通 Python 测试。

不得在本地：

- 下载或加载 Qwen；
- 执行 GPU 推理；
- 使用原始真实 TSV；
- 生成或提交真实工单模型输出。

### 16.2 服务器分阶段验收

1. **冒烟集**：少量人工挑选、跨主题脱敏工单，检查 schema、截断和明显类型错误；
2. **995 脱敏样本**：评估字段覆盖、warning/reject/repair 比例和 Prompt 边界；
3. **人工审计集**：按主题、长文本、复办、情绪和空间歧义分层抽样并标注；
4. **10 万规模运行**：仅在质量门槛通过后执行，检查吞吐、OOM、断点恢复和汇总质量；
5. **SAG 检索评估**：验证 seed recall、expansion precision 和错误超级节点。

### 16.3 核心质量指标

- event 当前事件忠实度；
- entity precision（按类型）；
- canonicalization 一致性；
- event-entity edge 准确性；
- intent/emotion/satisfaction/urgency 按字段准确性；
- evidence 可追溯率；
- seed recall；
- spatial expansion precision；
- JSON/schema 成功率、repair 率和 reject 率；
- orders/sec、tokens/sec、GPU 利用率和总耗时。

不得把“通过当前验证器的候选比例”称为实体准确率。

## 17. 服务器交付与运行方式

源码在本地完成静态开发后上传服务器运行。交付应包含：

- 明确的代码包或 Git commit；
- 服务器环境说明与依赖锁定；
- 模型路径、输入路径和输出路径通过环境变量/参数注入；
- 10/995/100k 等分阶段运行脚本；
- resume/retry/project-only 命令；
- 不泄露正文的日志与运行报告；
- 产物打包和回传检查脚本；
- GPU 运行前检查（CUDA、显存、模型目录、输入 schema、磁盘空间）；
- GPU 运行后检查（完成数、reject、截断、OOM、hash、文件行数）。

服务器默认命令必须消费脱敏多视图 JSONL，而非仓库内不存在的原始 TSV。真实路径不写死到源码。

## 18. Git 与隐私约束

允许提交并最终推送 GitHub：

- 源代码、配置和 Prompt 模板；
- 单元测试与人工构造的脱敏 fixture；
- 设计、实施和服务器运行文档；
- 不含真实工单文本的汇总报告模板。

禁止提交：

- 原始或真实脱敏工单数据；
- 模型权重；
- 工单级模型响应、rejects、links 或 discourse 产物；
- 可能含正文的日志；
- API key、服务器凭据和真实绝对路径。

最终推送前必须：

1. 检查 Git diff 和 staged files；
2. 确认用户现有换行符等无关工作区修改未混入；
3. 运行测试和秘密/敏感样式扫描；
4. 验证大文件和生成产物均被忽略；
5. 提交本功能代码与文档；
6. 经用户授权后推送当前 GitHub remote/branch，并返回网页 commit/branch 地址。

## 19. 实施边界与建议模块

为保持职责清晰，实施时建议拆分为：

- `work_order_input.py`：统一脱敏输入适配和稳定身份；
- `sag_semantic_prompt.py`：Prompt、few-shot 和输入预算；
- `sag_semantic_schema.py`：业务 schema、解析和枚举；
- `sag_semantic_validation.py`：确定性质量闸门；
- `sag_semantic_llm.py`：生成 backend、批处理、修复和运行编排；
- `sag_semantic_projection.py`：工单级记录到 SAG links/discourse；
- 现有 `sag_db.py`：加载投影结果并建立查询表；
- 服务器脚本：冒烟、样本、100k、resume、打包与报告。

具体文件名可在实施计划中依据现有代码结构微调，但必须保持输入、Prompt、验证、推理和投影的模块边界，避免继续把所有职责堆入 `sag_entity_llm.py`。

## 20. 已确认决策

- 采用方案 B：每条工单一次主推理，联合输出 event、entities 和 discourse；
- 工单级 JSONL 是权威中间产物，SAG entity links 是确定性查询投影；
- `problem_object`、`problem_behavior` 进行开放式识别；
- 第一阶段模型空间实体聚焦 road、intersection、POI；
- area/street 等 metadata 仍进入 SAG，但优先由程序确定性生成；
- discourse 作为 event attributes，不默认作为扩展实体；
- 删除模型自报 confidence，使用程序验证状态；
- 每条工单最多进行一次选择性修复；
- 保留完整脱敏 chunk 和完整 SAG 升级路径；
- 本地不运行模型；真实数据、GPU 推理和性能验收全部在服务器完成；
- 最终代码经验证后提交并推送 GitHub，但不上传任何真实工单或运行产物。
