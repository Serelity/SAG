# 对标 SAG 论文的纯 SAG-lite 工单检索实验设计

日期：2026-07-29

## 1. 背景与目标

当前项目已经完成一版 Hybrid Retrieval baseline，主要路径是：

```text
BM25 / Dense Retrieval
  -> 返回相似工单
  -> 按 metadata 做统计
```

这个 baseline 对“流动摆摊”“占道经营”这类语义主题已经有较好召回效果。下一阶段先不融合 Hybrid Retrieval，而是单独做一个“纯 SAG-lite”实验，用来验证 SAG 的结构化检索思想是否适合你的 12345 工单原始数据。

这里的 SAG 对标对象是论文和开源项目：

- 论文：SAG: SQL-Retrieval Augmented Generation with Query-Time Dynamic Hyperedges
- 代码：Zleap-AI/SAG-Benchmark
- 论文思想：把 chunk 转成语义完整的 event，并抽取 indexing entities；查询时通过 SQL join 动态连接共享实体的 events，形成局部动态 hyperedges。

本实验要回答的问题是：

```text
如果把每条工单视为一个 event，
再从 t_order_master.tsv 的结构化字段和 case_content 中抽取 entities，
仅靠 event-entity 索引、SQL join 和查询时动态扩展，
能否检索并统计“某区域/某街道/某道路附近是否有某类问题”的工单关系？
```

## 2. 与 SAG 论文的对标关系

SAG 论文和 Benchmark 项目强调以下结构：

```text
chunk -> event
chunk -> entities
event <-> entities
query-time SQL join expansion
Recall@K evaluation
```

本项目的对应关系：

| SAG 论文概念 | 本项目对应物 | 第一版实现方式 |
|---|---|---|
| chunk | 一条原始工单或一条工单中的核心诉求文本 | `t_order_master.tsv` 中的一行 |
| event | 一个语义完整的工单事件 | `case_content + case_goal + 时间 + 区域 + 分类` 组成 |
| entity | 索引实体 | 区、街道、道路、路口、POI、问题对象、问题行为、工单分类、月份 |
| latent hyperedge | 一个 event 连接多个 entities | `event_entity_links` 表中的一组 link |
| query-time dynamic hyperedge | 查询时由共享 entity 动态连接多个 event | SQL join 生成局部候选事件集合 |
| multi-hop expansion | 从种子 event 的实体继续扩展相关 event | 第一版支持 0-hop seed 和 1-hop expansion |
| Recall@K | 检索评估指标 | 构建弱标签和人工抽样标签两套评估 |

重要边界：

```text
纯 SAG-lite 不是完整复刻 SAG 论文。
第一版只验证 SQL event-entity 路径。
不使用向量召回、全文检索召回、LLM 抽取和 rerank。
```

这样做的原因是：我们现在要隔离验证“结构化事件-实体关系”本身的价值。如果一开始混入 Hybrid Retrieval，就看不清 SAG 对字段缺失和空间关系问题到底贡献了多少。

## 3. 原始数据对标

本实验以 `t_order_master.tsv` 作为事实源，不把 multiview JSONL 当成唯一输入。multiview JSONL 可以继续作为上游清洗产物，但 SAG-lite 的字段设计必须能追溯到原始 TSV 字段。

原始 TSV 关键字段分组如下。

### 3.1 工单标识字段

```text
id
process_instance_id
order_no
order_id
thrid_order_id
processInstanceId
dispatchOrderId
recordId
```

用途：

```text
生成稳定 doc_id / event_id。
保留原始业务编号，便于回查。
不直接暴露敏感编号到最终报告。
```

### 3.2 核心语义字段

```text
title
case_content
case_goal
case_labels
case_accord_ext
remark
custom_form_data_str
visitAdvContent
appeal_result_opinions
```

第一版核心字段：

```text
case_content
case_goal
title
```

其中 `case_content` 权重最高，因为它最直接描述群众反映的问题、地点、对象和行为。

### 3.3 空间字段

```text
area_code_city
area_code_area
area_code_street
address_detail
area_code
toAreaCode
case_lnglat
belong_dept
deptName
orgName
notice_org_ids
appeal_dept
```

第一版直接使用：

```text
area_code_city
area_code_area
area_code_street
address_detail
case_lnglat
```

注意：

```text
area_code_street 缺失时，不能直接判定工单没有街道/道路信息。
case_content 和 address_detail 可能包含更细空间信息。
case_lnglat 如果有值，后续可进入真实空间半径查询；第一版只做字段保留和覆盖率统计。
```

### 3.4 时间字段

```text
call_time
create_date
update_date
plan_finish_time
case_complete_time
plan_sign_time
callEndTime
order_finish_time
appeal_feedback_date
change_time
```

第一版主时间字段：

```text
call_time
```

派生字段：

```text
call_month = call_time[:7]
```

### 3.5 分类字段

```text
case_type
case_accord_type_one_name
case_accord_type_two_name
case_accord_type_three_name
case_accord_type_four_name
case_accord_type_five_name
case_accord_code
first_level_affiliation
second_level_affiliation
third_level_affiliation
fourth_level_affiliation
fifth_level_affiliation
sixth_level_affiliation
seventh_level_affiliation
```

第一版直接使用：

```text
case_accord_type_one_name   -> type1
case_accord_type_two_name   -> type2
case_accord_type_three_name -> type3
case_accord_type_four_name  -> type4
case_accord_type_five_name  -> type5
case_accord_code
```

用途：

```text
作为 case_type entity。
辅助判断问题对象和问题行为。
评估 SAG 检索结果是否落在合理业务类别。
```

### 3.6 处理状态与反馈字段

```text
order_status
first_order_status
secord_order_status
atomic_order_status
case_solve
contact_timely
resultSatisfied
firstVistSatisfied
firstVisitSatisfied
visitResult
appeal_status
appeal_dept
```

第一版只保留，不作为检索条件。后续可用于分析：

```text
某类问题是否高频反复投诉
某类问题办结满意度
某类问题是否集中在特定区域
```

### 3.7 敏感字段

```text
call_number
contact_number
customerName
customerSex
orderUserName
orderUserSex
orderUserPhone2
creator_id
updator_id
tenant_id
```

处理规则：

```text
不进入检索文本。
不进入最终报告。
如必须保留统计，先脱敏或哈希。
```

## 4. 数据结构设计

第一版使用 DuckDB，便于在服务器无 Docker 环境下直接运行。

### 4.1 source_orders

保存从原始 TSV 读取后的核心字段。该表是事实源快照，不做语义推断。

字段：

```text
doc_id
raw_id_hash
order_id_hash
title_clean
case_content_clean
case_goal_clean
address_detail_clean
call_time
call_month
area_code_city
area_code_area
area_code_street
case_lnglat
type1
type2
type3
type4
type5
case_accord_code
order_source
order_type
order_status
service_object_type
```

### 4.2 sag_events

一条工单第一版对应一个 event。

字段：

```text
event_id
doc_id
event_text
event_time
event_month
event_source
event_status
```

第一版 `event_text` 由以下字段组成：

```text
title_clean
case_content_clean
case_goal_clean
address_detail_clean
type1/type2/type3/type4/type5
area_code_city/area_code_area/area_code_street
call_time
```

注意：`event_text` 用于保存事件完整语义，不用于 embedding。

### 4.3 sag_entities

保存实体字典。相同标准化实体只保存一次。

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
department
lnglat
```

### 4.4 sag_event_entity_links

保存 event 和 entity 的多对多关系。

字段：

```text
event_id
doc_id
entity_id
entity_type
entity_value
source_field
source_channel
confidence
matched_text
```

`source_channel` 取值：

```text
metadata
case_content
case_goal
title
address_detail
rule
```

置信度第一版规则：

```text
1.00  来自原始 TSV 明确结构化字段
0.90  来自 case_content/address_detail 中明确地点表达
0.80  来自 title/case_goal 中明确问题表达
0.70  来自关键词词典匹配的问题对象或问题行为
0.60  来自弱规则 POI 短语
```

### 4.5 sag_query_runs

保存每次查询实验的配置和摘要。

字段：

```text
run_id
query_name
query_config_json
created_at
matched_orders
elapsed_ms
```

### 4.6 sag_query_results

保存每次查询命中的 event 及其解释路径。

字段：

```text
run_id
rank
doc_id
event_id
score
match_stage
matched_entities_json
explanation_json
```

`match_stage` 取值：

```text
seed_entity
one_hop_expansion
```

## 5. 实体抽取规则

第一版使用轻量规则，不追求完整中文 NER。目标是先建立可解释、可评估、可复现的 SAG baseline。

### 5.1 metadata 实体

直接从原始字段生成：

```text
call_month                         -> time_month
area_code_area                     -> area
area_code_street                   -> street
case_lnglat                        -> lnglat
belong_dept/deptName/orgName       -> department
case_accord_type_*_name            -> case_type
```

### 5.2 文本空间实体

从 `case_content_clean`、`address_detail_clean`、`title_clean` 中抽取：

```text
区县：
  钟楼区、天宁区、新北区、武进区、金坛区、溧阳市、常州市经济开发区、市本级等。

街道/镇：
  xx街道、xx镇。

道路：
  xx路、xx街、xx大道、xx巷、xx弄、xx桥、xx线。

路口：
  xx路和xx路交叉口、xx路与xx路交界处、xx路口、xx附近。

POI：
  xx小区、xx市场、xx学校、xx广场、xx商场、xx夜市、xx公园、xx医院、xx菜场、xx地铁站。
```

### 5.3 问题实体

第一版围绕“流动摆摊/占道经营”主题建立可解释词典：

```text
problem_object:
  流动摊贩
  游商摊贩
  摊贩
  夜市摊贩
  小摊
  商贩

problem_behavior:
  摆摊
  设摊
  占道经营
  无照经营
  店外经营
  影响通行
  扰民
  油烟
```

后续可以扩展为通用实体抽取或 LLM 抽取，但第一版必须保持规则可解释。

## 6. 纯 SAG 查询流程

纯 SAG-lite 的查询不使用 embedding，也不使用 BM25。查询配置显式指定实体条件。

输入配置示例：

```json
{
  "query_name": "stall",
  "seed_entities": [
    {
      "entity_type": "problem_object",
      "values": ["流动摊贩", "游商摊贩", "摊贩", "商贩"],
      "operator": "OR"
    },
    {
      "entity_type": "problem_behavior",
      "values": ["摆摊", "设摊", "占道经营", "无照经营", "店外经营"],
      "operator": "OR"
    }
  ],
  "seed_group_operator": "AND",
  "space_entities": [],
  "filters": {
    "call_month_gte": "2024-01",
    "call_month_lte": "2024-12"
  },
  "expansion": {
    "enabled": true,
    "max_hops": 1,
    "frontier_entity_types": ["area", "street", "road", "intersection", "poi"],
    "max_expanded_events": 2000
  },
  "representative_limit": 10
}
```

### 6.1 Seed retrieval

先根据查询实体找到种子 events。

规则：

```text
同一个 entity group 内部使用 OR。
不同 entity group 之间使用 AND。
时间过滤应用在 event_month。
```

例子：

```text
problem_object in (流动摊贩, 游商摊贩, 摊贩, 商贩)
AND
problem_behavior in (摆摊, 设摊, 占道经营, 无照经营, 店外经营)
AND
event_month between 2024-01 and 2024-12
```

### 6.2 Query-time dynamic hyperedge expansion

这是对标 SAG 的关键步骤。

每个 seed event 都连接多个 entities，例如：

```text
event_A:
  problem_object=流动摊贩
  problem_behavior=占道经营
  area=钟楼区
  street=永红街道
  road=广成路
  road=江春路
```

查询时把这些空间实体作为 frontier，通过 SQL join 找到共享这些实体的其他 events：

```text
seed event
  -> frontier entities: area/street/road/intersection/poi
  -> shared-entity events
```

这一步形成局部动态关系，不提前构建全局知识图谱。

第一版只做 1-hop expansion，避免召回无限扩散：

```text
0-hop: 直接满足问题实体的工单。
1-hop: 与 0-hop 工单共享空间实体的其他工单。
```

### 6.3 Scoring

纯 SAG-lite 不使用语义相似度，分数来自结构证据：

```text
score =
  matched_seed_entity_count * 10
  + matched_space_entity_count * 3
  + source_confidence_sum
  - expansion_penalty
```

其中：

```text
seed_entity 命中优先于 one_hop_expansion。
metadata 与 case_content 同时支持的实体加分。
只通过 area 共享的 expansion 权重较低。
通过 street/road/intersection/poi 共享的 expansion 权重较高。
```

这个分数不是最终质量指标，只用于结果排序。

## 7. 输出报告

纯 SAG 查询输出：

```text
query
matched_orders
seed_orders
expanded_orders
statistics.by_month
statistics.by_area_metadata
statistics.by_area_entity
statistics.by_street_metadata
statistics.by_street_entity
statistics.by_road_entity
statistics.by_intersection_entity
statistics.by_poi_entity
statistics.by_problem_object
statistics.by_problem_behavior
entity_coverage
metadata_recovery
conflict_report
representative_cases
retrieval
evaluation
```

### 7.1 entity_coverage

必须包含：

```text
area
street
road
intersection
poi
problem_object
problem_behavior
case_type
lnglat
```

每个实体类型报告：

```text
events_total
events_with_entity
coverage
source_breakdown
```

### 7.2 metadata_recovery

用于回答字段缺失问题：

```text
metadata_street_missing
metadata_street_missing_but_text_street_found
metadata_street_missing_but_text_road_found
metadata_street_missing_but_text_intersection_found
metadata_street_missing_but_text_poi_found
recovery_rate
```

这个指标是本阶段核心，因为它能直接说明：

```text
街道字段缺失时，case_content/address_detail 能补回多少空间检索能力。
```

### 7.3 conflict_report

用于记录结构化字段和文本内容的冲突：

```text
metadata_area
text_area
metadata_street
text_street
conflict_count
examples
```

第一版不自动裁决冲突，只报告冲突。

## 8. 结果评估方案

这部分必须进入实现范围。否则只看命中数量，无法判断纯 SAG 是否有效。

### 8.1 与 SAG 论文指标的对标

SAG Benchmark 使用 Recall@1 / Recall@2 / Recall@5 / Recall@10 等指标评估多跳检索结果。我们的数据没有标准 QA gold answers，因此不能直接照搬 HotpotQA/2WikiMultiHopQA/MuSiQue 的评估方式，但要保留同类思想：

```text
给定一个查询，
系统返回 topK 工单，
判断 topK 中是否覆盖应命中的工单集合。
```

第一版使用三类评估。

### 8.2 弱标签评估

弱标签不是绝对真值，只用于自动化粗评。

以“流动摆摊/占道经营”为例，构建 weak_gold：

```text
type3 in (无照经营游商, 店外经营, 无证照餐饮店)
OR
case_content/title/case_goal 包含：
  流动摊贩、游商摊贩、摆摊、设摊、占道经营、无照经营、店外经营
```

指标：

```text
weak_precision@10
weak_precision@50
weak_precision@100
weak_recall@100
weak_recall@500
weak_recall@1000
```

解释：

```text
Precision@K：前 K 条中有多少比例符合弱标签。
Recall@K：弱标签集合中有多少比例被前 K 条覆盖。
```

### 8.3 人工抽样评估

弱标签会有偏差，所以必须保留人工评估入口。

第一版输出一个待标注文件：

```text
outputs/sag_lite.eval_samples.stall.100k.jsonl
```

每条样本包含：

```text
doc_id
rank
match_stage
score
case_content
case_goal
metadata_area
metadata_street
matched_entities
explanation
label
label_reason
```

人工标注规则：

```text
2 = 高度相关，明确是该主题问题。
1 = 部分相关，属于附近/类似治理问题，但主题不完全一致。
0 = 不相关。
```

人工评估指标：

```text
manual_precision@10
manual_precision@50
manual_precision@100
mean_label_score@10
mean_label_score@50
```

### 8.4 实体抽取质量评估

为了判断 SAG 的根基是否可靠，需要评估实体抽取本身。

第一版输出实体抽样：

```text
outputs/sag_lite.entity_eval_samples.100k.jsonl
```

人工检查：

```text
地点实体是否抽对。
问题对象是否抽对。
问题行为是否抽对。
metadata 和文本来源是否区分正确。
```

指标：

```text
entity_precision_by_type
space_entity_precision
problem_entity_precision
metadata_vs_text_conflict_rate
```

### 8.5 空间补全评估

专门评估字段缺失时的空间能力：

```text
street_metadata_coverage
street_entity_coverage
road_entity_coverage
intersection_entity_coverage
poi_entity_coverage
metadata_missing_recovery_rate
```

重点看：

```text
area_code_street 为空的工单中，
有多少能从 case_content/address_detail 抽到道路、路口、POI 或街道。
```

### 8.6 稳定性评估

同一个查询改变 expansion 参数，观察统计是否稳定：

```text
max_hops = 0 vs 1
max_expanded_events = 500 vs 1000 vs 2000
frontier_entity_types = street/road/intersection/poi 是否包含 area
```

输出：

```text
matched_orders_delta
top_area_distribution_delta
top_street_distribution_delta
representative_case_overlap
```

如果加入 area 后结果大幅膨胀，说明 area 粒度过粗，应降低 area expansion 权重或默认不作为 expansion frontier。

### 8.7 成本评估

记录：

```text
build_seconds
query_elapsed_ms
events_count
entities_count
links_count
duckdb_size_mb
```

这能判断纯 SAG-lite 是否适合后续扩展到百万级数据。

## 9. 与 Hybrid Retrieval baseline 的关系

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
metadata 缺失被 case_content/address_detail 补充的比例
weak_precision@K
weak_recall@K
manual_precision@K
query_elapsed_ms
```

## 10. 成功标准

第一版成功标准：

- 能从 `t_order_master.tsv` 或 100k multiview JSONL 构建 DuckDB。
- 能保留并映射原始 TSV 的关键字段。
- 能抽取 metadata、case_content、address_detail、title 中的基础实体。
- 能运行“流动摆摊/占道经营”纯 SAG 查询。
- 能执行 0-hop seed retrieval 和 1-hop SQL expansion。
- 输出结果不依赖 embedding、BM25 和 LLM。
- 能明确显示 metadata 街道覆盖率与文本实体补充覆盖率。
- 能输出弱标签评估、人工评估样本、实体抽取评估样本。
- 能生成可与当前 Hybrid Retrieval baseline 对比的 JSON 报告。

## 11. 实现文件范围

计划新增：

```text
src/ragflow_style_pipeline/sag_entities.py
src/ragflow_style_pipeline/sag_db.py
src/ragflow_style_pipeline/sag_query.py
src/ragflow_style_pipeline/sag_eval.py
tests/test_sag_entities.py
tests/test_sag_db.py
tests/test_sag_query.py
tests/test_sag_eval.py
configs/sag_query_stall.json
scripts/build_sag_lite_100k.sh
scripts/query_sag_lite_stall_100k.sh
scripts/evaluate_sag_lite_stall_100k.sh
docs/11-纯SAG工单检索实验.md
```

尽量不改动现有 Hybrid Retrieval 代码，避免 baseline 被污染。

## 12. 风险与限制

- 规则抽取会漏掉复杂地名和口语化位置。
- 没有经纬度时，“附近”只能先做文本空间实体近似，不能做真实半径查询。
- 问题实体词典第一版覆盖“流动摆摊/占道经营”，还不是通用事件抽取系统。
- 如果 case_content 中地点与 metadata 区域冲突，第一版只记录两个来源，不自动裁决。
- 弱标签评估不是绝对真值，必须结合人工抽样评估。
- 纯 SAG 查询召回可能比 Hybrid Retrieval 少，但它能更清楚解释“为什么这些工单有关联”。

## 13. 下一步

设计确认后，按 TDD 实现：

```text
1. 先写实体抽取测试。
2. 再实现 sag_entities.py。
3. 写 DuckDB 建库测试。
4. 再实现 sag_db.py。
5. 写查询统计测试。
6. 再实现 sag_query.py。
7. 写评估指标测试。
8. 再实现 sag_eval.py。
9. 本地跑测试。
10. 打包给服务器运行。
```

## 14. 参考资料

- SAG 论文页：https://huggingface.co/papers/2606.15971
- SAG Benchmark 官方仓库：https://github.com/Zleap-AI/SAG-Benchmark
