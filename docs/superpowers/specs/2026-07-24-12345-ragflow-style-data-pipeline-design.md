# 12345 RAGFlow-style 数据管道设计

## 目标

参考 RAGFlow 的“文档解析、规范化、元数据建模、索引准备”思想，为 `G:\12345_pro_promax\data\t_order_master.tsv` 构建一个可复现的 RAG 数据预处理层。

## 背景

源数据是 12345 政务热线工单主表，约 1.25GB、982,435 行、129 列。它不是普通作文文本，而是一行一个业务工单。第一版应把每行工单转换成一个脱敏后的 RAG 文档，并保留地区、分类、时间等 metadata。

## 设计范围

包含：

- 流式读取 TSV。
- 识别字段数异常行。
- 选择核心字段。
- 对正文中的手机号、身份证号等做脱敏。
- 生成 `doc_id`、`text`、`metadata`。
- 输出 JSONL。
- 输出质量报告。

不包含：

- LLM 调用。
- Embedding 生成。
- RAGFlow 源码修改。
- 原始 TSV 改写。
- 原始敏感数据输出。

## 数据模型

每行输出结构：

```json
{
  "doc_id": "工单主键",
  "text": "规范化中文文档正文",
  "metadata": {
    "order_id": "工单编号",
    "service_object_type": "诉求类型",
    "area_code_city": "城市",
    "area_code_area": "区县",
    "area_code_street": "街道",
    "type1": "一级分类",
    "type2": "二级分类",
    "type3": "三级分类",
    "order_source": "来源渠道",
    "order_type": "工单类型",
    "order_status": "状态",
    "call_time": "来电时间",
    "call_month": "YYYY-MM"
  }
}
```

## 字段策略

正文由 `case_content`、`case_goal`、诉求类型、业务分类、所属区域、时间和来源拼接。

metadata 保留 `id`、`order_id`、地区、分类、来源、类型、状态、时间等可过滤字段。

敏感字段如电话号码、身份证号、姓名、录音 ID、附件 ID 第一版不输出。正文内检测到手机号或身份证号时替换成 `[手机号]`、`[身份证号]`。

## 处理策略

- 使用 Python 标准库 `csv` 流式读取，避免一次性加载 1.25GB 文件。
- 默认只接受字段数等于表头字段数的行。
- 字段数异常的行写入质量报告，不写入主 JSONL。
- 空值、`NULL`、`null` 统一视为空。
- 文本字段逐项拼接，空字段跳过。
- 输出 UTF-8 JSONL。

## 工程边界

脚本放在 `src/ragflow_style_pipeline/`。

配置放在 `configs/t_order_master_schema.yaml`，但为避免额外依赖，第一版可使用简单 YAML 子集或 JSON 配置。如果 `cz12345` 环境没有 YAML 解析库，则改用 JSON 配置。

输出放在 `outputs/`，该目录应被 `.gitignore` 忽略，避免提交数据样本。

## 验证方式

- 用前 1000 行生成小样本。
- 检查输出每行都是合法 JSON。
- 检查 `doc_id`、`text`、`metadata` 必填。
- 检查输出正文没有手机号和身份证号模式。
- 检查质量报告包含读取行数、输出行数、跳过行数、脱敏命中数。

## 后续扩展

第一版稳定后再做：

- 超长工单切分。
- `knowledge_quote` JSON 解析。
- 分类体系统计。
- Elasticsearch 写入。
- Embedding 写入。
- 与 RAGFlow HTTP API 或数据库导入流程对接。
