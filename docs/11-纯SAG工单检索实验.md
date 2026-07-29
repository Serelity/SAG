# 11-纯 SAG 工单检索实验

日期：2026-07-29

## 1. 为什么先做纯 SAG

当前项目已经有一个 Hybrid Retrieval baseline：

```text
BM25 / Dense Retrieval
  -> 找相似工单
  -> 按 metadata 统计
```

这一版 baseline 可以继续保留，用来对比。

现在单独做纯 SAG，是为了回答另一个问题：

```text
如果不用 embedding、不用 BM25、不用 LLM，
只把工单拆成 event 和 entity，
再用 SQL join 做关系检索，
效果到底怎么样？
```

这样可以看清楚 SAG 思路本身的贡献，不会被向量检索效果掩盖。

## 2. 纯 SAG-lite 和 SAG 论文怎么对应

SAG 论文的关键思想可以简化为：

```text
chunk -> event
chunk -> entities
event <-> entities
query-time SQL join expansion
```

放到你的 12345 工单里：

```text
一行 t_order_master.tsv = 一个 chunk
一条工单事件 = 一个 event
从工单里抽出来的区、街道、道路、问题对象、问题行为 = entities
一条工单连接多个实体 = latent hyperedge
查询时通过共享实体连接相关工单 = query-time dynamic hyperedge
```

## 3. 哪些字段进入 event

第一版 event 主要由这些字段组成：

```text
title
case_content
case_goal
address_detail
area_code_city
area_code_area
area_code_street
case_accord_type_one_name
case_accord_type_two_name
case_accord_type_three_name
case_accord_type_four_name
case_accord_type_five_name
call_time
```

其中最重要的是：

```text
case_content
```

因为它最直接描述：

```text
群众反映了什么问题
问题发生在哪里
涉及什么对象
具体行为是什么
```

## 4. 哪些字段进入 entity

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

其中空间实体包括：

```text
area
street
road
intersection
poi
lnglat
```

问题实体包括：

```text
problem_object
problem_behavior
case_type
```

## 5. 为什么 case_content 是核心

你的数据里 `area_code_street` 可能缺失，但 `case_content` 里经常会写：

```text
钟楼区某某路
某某街道
某某路和某某路交叉口
某某小区门口
某某市场附近
```

所以字段缺失不代表空间信息不存在。

纯 SAG-lite 的关键就是：

```text
metadata 有值就用 metadata。
metadata 缺失时，从 case_content / address_detail 里抽空间实体。
原始 metadata 不改。
文本抽出来的实体单独记录来源和置信度。
```

## 6. 为什么不能直接用街道字段硬过滤

如果 `area_code_street` 覆盖率很低，直接用它过滤会漏掉大量工单。

例如：

```text
area_code_street = 空
case_content = “钟楼区茶花路和白杨路交叉口有流动摊贩”
```

这条工单虽然没有街道字段，但它有道路和路口信息。  
纯 SAG-lite 会把它抽成：

```text
area = 钟楼区
road = 茶花路
road = 白杨路
intersection = 茶花路和白杨路交叉口
problem_object = 流动摊贩
```

## 7. 服务器运行前准备

进入项目目录：

```bash
cd ragflow-learning-plan
```

含义：

```text
进入服务器上的项目文件夹。
后面的脚本都要在这个目录下运行。
```

把原始 TSV 放到项目 data 目录：

```bash
mkdir -p data
cp /你的实际路径/t_order_master.tsv data/t_order_master.tsv
```

含义：

```text
mkdir -p data：
  如果 data 文件夹不存在，就创建它。

cp /你的实际路径/t_order_master.tsv data/t_order_master.tsv：
  把服务器上真实的原始 TSV 文件复制到项目默认读取位置。
  左边路径要换成你的真实文件路径。
```

如果服务器还没有 DuckDB：

```bash
pip install duckdb
```

含义：

```text
给当前 Python / conda 环境安装 DuckDB。
DuckDB 是一个本地分析数据库，这里用它保存 event、entity 和 link 表。
```

## 8. 运行建库脚本

```bash
bash scripts/build_sag_lite_100k.sh
```

含义：

```text
bash：
  用 Bash 执行脚本。

scripts/build_sag_lite_100k.sh：
  建立纯 SAG-lite 数据库。

默认读取：
  data/t_order_master.tsv

默认输出：
  outputs/sag_lite.100k.duckdb

默认行数：
  前 100000 行。
```

如果想改行数：

```bash
LIMIT=200000 bash scripts/build_sag_lite_100k.sh
```

含义：

```text
LIMIT=200000：
  临时把读取行数改成 200000。
```

## 9. 运行纯 SAG 查询

```bash
bash scripts/query_sag_lite_stall_100k.sh
```

含义：

```text
运行“流动摆摊 / 占道经营”纯 SAG 查询。
这一步不使用 embedding。
这一步不使用 BM25。
这一步不调用 LLM。
它只通过 event-entity 关系和 SQL join 找工单。
```

默认配置：

```text
configs/sag_query_stall.json
```

默认输出：

```text
outputs/sag_lite.query.stall.100k.json
```

## 10. 运行评估脚本

```bash
bash scripts/evaluate_sag_lite_stall_100k.sh
```

含义：

```text
根据查询结果生成评估文件。
包括人工标注样本和实体抽取检查样本。
```

默认输出：

```text
outputs/sag_lite.eval_samples.stall.100k.jsonl
outputs/sag_lite.entity_eval_samples.100k.jsonl
```

## 11. 怎么读结果

核心结果文件：

```text
outputs/sag_lite.query.stall.100k.json
```

重点看这些字段：

```text
matched_orders：
  总共命中多少工单。

seed_orders：
  直接命中“流动摊贩 + 占道经营”等问题实体的工单数。

expanded_orders：
  通过共享街道、道路、路口、POI 扩展出来的工单数。

metadata_recovery：
  街道字段缺失时，case_content/address_detail 能补回多少空间信息。

entity_coverage：
  每种实体类型覆盖率。

evaluation：
  weak_precision@K 和 weak_recall@K。

representative_cases：
  排名前面的代表工单，方便人工看质量。
```

## 12. 当前限制

第一版有几个明确限制：

```text
1. 实体抽取是规则版，不是完整中文 NER。
2. “附近”暂时不是经纬度半径查询，而是共享道路/街道/POI 的近似关系。
3. 弱标签评估不是绝对真值，需要人工抽样一起看。
4. 只做纯 SAG，不和 Hybrid Retrieval 融合。
```

## 13. 下一步

服务器跑完后，把这些文件发回来：

```text
outputs/sag_lite.query.stall.100k.json
outputs/sag_lite.eval_samples.stall.100k.jsonl
outputs/sag_lite.entity_eval_samples.100k.jsonl
```

我会帮你分析：

```text
纯 SAG 的命中质量
空间实体补全能力
seed 和 expansion 哪个贡献更大
是否应该降低或提高某些实体权重
下一步是否进入 SAG + Hybrid Retrieval 融合
```
