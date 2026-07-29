# 纯 SAG-lite 工单检索实验设计

日期：2026-07-29

## 1. 背景与目标

当前项目已经完成一版 Hybrid Retrieval baseline，主要路径是：

```text
BM25 / Dense Retrieval
  -> 返回相似工单
  -> 按 metadata 做统计
```

这个 baseline 对“流动摆摊”“占道经营”这类语义主题已经有较好召回效果，但它仍然主要回答“哪些工单语义相似”。下一阶段要单独验证 SAG 思路是否有价值，因此本实验先实现一个不依赖 embedding、不融合 Hybrid Retrieval、不依赖 LLM 的纯 SAG-lite baseline。

本实验要回答的问题是：

```text
如果把每条工单视为一个 event，
再从 metadata 和 case_content 中抽取实体 entity，
仅靠 event-entity 关系和 SQL 查询，
能否检索并统计类似“某街道附近是否有流动摆摊问题”的工单关系？
```

## 2. 非目标

第一版明确不做以下内容：

- 不调用 LLM 做查询理解或实体抽取。
- 不使用 BGE-M3、BM25 或其他语义检索结果。
- 不做 Hybrid Retrieval 与 SAG 的融合。
- 不接入图数据库。
- 不做真实地理编码和经纬度半径查询。
- 不把从文本推断出的实体回写到原始 metadata 字段。

这些能力可以作为后续阶段，但不进入本次纯 SAG-lite baseline。

## 3. 核心设计

纯 SAG-lite 使用三类核心对象：

```text
event
  一条工单对应一个事件。

entity
  从工单 metadata 或 case_content 中抽取出的实体。

event_entity_link
  工单事件和实体之间的多对多关系。
```

设计原则：

- `case_content` 是核心语义来源，也是补充空间实体的重要来源。
- `metadata` 是结构化辅助来源，但低覆盖字段不能作为硬过滤的唯一依据。
- 从 `case_content` 抽取出的字段放在实体表中，并记录来源为 `case_content`。
- 原始 metadata 保持不变。
- 统计结果必须报告实体覆盖率，特别是街道、道路、路口等空间实体覆盖率。

## 4. 数据表设计

第一版使用 DuckDB，便于在服务器无 Docker 环境下直接运行。

### 4.1 work_order_events

一条工单一行。

字段：

```text
doc_id
case_content_clean
case_goal_clean
display_text
call_time
call_month
area_code_city
area_code_area
area_code_street
type1
type2
type3
order_source
order_type
order_status
service_object_type
```

### 4.2 work_order_entities

保存实体字典。相同实体只保存一次。

字段：

```text
entity_id
entity_type
entity_value
normalized_value
```

第一版实体类型：

```text
time_month
area
street
road
intersection
poi
problem_object
problem_behavior
case_type
```

### 4.3 event_entity_links

保存工单和实体之间的关系。

字段：

```text
doc_id
entity_id
entity_type
entity_value
source
confidence
```

`source` 取值：

```text
metadata
case_content
case_goal
rule
```

`confidence` 第一版只使用规则置信度：

```text
1.0  来自 metadata 的明确字段
0.9  来自 case_content 的明确行政区、街道、道路、路口
0.7  来自关键词规则的问题对象或问题行为
```

## 5. 实体抽取规则

第一版使用轻量规则，不追求完整中文 NER。

### 5.1 metadata 实体

直接从 metadata 生成：

```text
call_month        -> time_month
area_code_area    -> area
area_code_street  -> street
type3             -> case_type
```

### 5.2 case_content 空间实体

从 `case_content_clean` 中抽取：

```text
区县：包含“区”“市本级”“经开区”等常见行政区域词。
街道/镇：匹配“xx街道”“xx镇”。
道路：匹配“xx路”“xx街”“xx大道”“xx巷”“xx弄”。
路口：匹配“xx路和xx路交叉口”“xx路与xx路交界处”等表达。
POI：第一版只保留粗规则，例如“市场”“学校”“小区”“广场”“商场”“夜市”等后缀短语。
```

### 5.3 问题实体

第一版围绕“流动摆摊/占道经营”主题建立可解释规则：

```text
problem_object:
  流动摊贩
  游商摊贩
  摊贩
  夜市摊贩

problem_behavior:
  摆摊
  占道经营
  无照经营
  店外经营
  影响通行
```

后续可以扩展为通用词典或 LLM 抽取，但第一版先保持可控。

## 6. 纯 SAG 查询流程

输入配置示例：

```json
{
  "query_name": "stall",
  "required_entities": [
    {"entity_type": "problem_object", "values": ["流动摊贩", "游商摊贩", "摊贩"]},
    {"entity_type": "problem_behavior", "values": ["摆摊", "占道经营", "无照经营"]}
  ],
  "optional_entities": [],
  "filters": {
    "call_month_gte": "2024-01",
    "call_month_lte": "2024-12"
  },
  "representative_limit": 10
}
```

查询逻辑：

```text
1. 根据 required_entities 找到候选 doc_id。
2. 同一实体组内部使用 OR。
3. 不同实体组之间使用 AND。
4. 应用时间过滤。
5. 对结果做统计聚合。
6. 输出代表工单和实体覆盖率。
```

例如：

```text
problem_object in (流动摊贩, 游商摊贩, 摊贩)
AND
problem_behavior in (摆摊, 占道经营, 无照经营)
AND
call_month between 2024-01 and 2024-12
```

## 7. 输出报告

纯 SAG 查询输出：

```text
query
matched_orders
statistics.by_month
statistics.by_area
statistics.by_street_metadata
statistics.by_street_entity
statistics.by_road_entity
statistics.by_problem_object
statistics.by_problem_behavior
entity_coverage
representative_cases
retrieval
```

其中 `entity_coverage` 必须包含：

```text
area
street
road
intersection
poi
problem_object
problem_behavior
```

街道相关统计必须区分：

```text
metadata 街道统计
case_content 抽取街道统计
合并实体街道统计
```

这样可以直接观察字段缺失时，文本实体是否补回了空间粒度。

## 8. 与 Hybrid Retrieval baseline 的关系

本实验只输出纯 SAG-lite 结果，不做融合。

后续对比时使用两份结果：

```text
Hybrid baseline:
  topic_analysis.stall.100k.json

Pure SAG-lite:
  sag_lite.query.stall.100k.json
```

对比维度：

```text
matched_orders
代表工单相关性
月份分布
区域分布
街道覆盖率
道路/路口覆盖率
metadata 缺失被 case_content 补充的比例
```

## 9. 成功标准

第一版成功标准：

- 能从 100k multiview JSONL 构建 DuckDB。
- 能抽取 metadata 和 case_content 中的基础实体。
- 能运行“流动摆摊/占道经营”纯 SAG 查询。
- 输出结果不依赖 embedding 和 LLM。
- 能明确显示 metadata 街道覆盖率与文本实体补充覆盖率。
- 能生成可与当前 Hybrid Retrieval baseline 对比的 JSON 报告。

## 10. 实现文件范围

计划新增：

```text
src/ragflow_style_pipeline/sag_entities.py
src/ragflow_style_pipeline/sag_db.py
src/ragflow_style_pipeline/sag_query.py
tests/test_sag_entities.py
tests/test_sag_db.py
tests/test_sag_query.py
configs/sag_query_stall.json
scripts/build_sag_lite_100k.sh
scripts/query_sag_lite_stall_100k.sh
docs/11-纯SAG工单检索实验.md
```

尽量不改动现有 Hybrid Retrieval 代码，避免 baseline 被污染。

## 11. 风险与限制

- 规则抽取会漏掉复杂地名和口语化位置。
- 没有经纬度时，“附近”只能先做文本空间实体近似，不能做真实半径查询。
- 问题实体词典第一版覆盖“流动摆摊/占道经营”，还不是通用事件抽取系统。
- 如果 case_content 中地点与 metadata 区域冲突，第一版只记录两个来源，不自动裁决。
- 纯 SAG 查询召回可能比 Hybrid Retrieval 少，但它能更清楚解释“为什么这些工单有关联”。

## 12. 下一步

设计确认后，按 TDD 实现：

```text
1. 先写实体抽取测试。
2. 再实现 sag_entities.py。
3. 写 DuckDB 建库测试。
4. 再实现 sag_db.py。
5. 写查询统计测试。
6. 再实现 sag_query.py。
7. 本地跑测试。
8. 打包给服务器运行。
```
