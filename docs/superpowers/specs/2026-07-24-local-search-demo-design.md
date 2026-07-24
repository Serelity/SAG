# 本地检索 Demo 设计说明

## 目标

基于已经生成的脱敏 JSONL 工单文档，先实现一个不依赖外部模型、不依赖向量数据库的本地检索 demo，用 1000 样本和 10 万样本分别验证“工单行转文本”之后是否具备基本可检索性。

## 为什么先做本地检索

本地检索是 RAG 系统的基线层。它不解决最终的语义理解问题，但能快速暴露三个关键问题：

- 文档文本是否包含用户会查询的关键词。
- metadata 字段是否能支撑地区、分类、月份等过滤。
- 1000 到 10 万规模下，数据格式和索引逻辑是否稳定。

如果这一步检索结果很差，直接上 embedding 或向量库也很难得到稳定效果，因为底层文档建模还没有被验证。

## 设计方案

第一版采用标准库实现的 BM25 风格关键词检索：

```text
JSONL 文档
  -> 加载 doc_id / text / metadata
  -> 中文 bigram + 英文/数字 token 化
  -> 建立内存倒排索引
  -> 根据查询词计算 BM25 风格分数
  -> 可选 metadata 过滤
  -> 输出 Top-K 检索结果
```

## 模块边界

- `text_tokenizer.py`：负责把文本切成 token。中文用相邻两个字组成 bigram，英文和数字按连续片段切分。
- `local_search.py`：负责加载 JSONL、建立倒排索引、执行 BM25 风格检索。
- `search_jsonl.py`：负责命令行交互，读取参数并打印检索结果。

## 支持的过滤字段

第一版只支持常用结构化过滤：

- `--area`：对应 `metadata.area_code_area`
- `--type1`：对应 `metadata.type1`
- `--type2`：对应 `metadata.type2`
- `--type3`：对应 `metadata.type3`
- `--month`：对应 `metadata.call_month`

## 测试范围

自动化测试覆盖：

- 中文、英文、数字 token 化。
- BM25 检索能把更相关文档排在前面。
- metadata 过滤能缩小候选文档范围。
- CLI 能读取 JSONL 并输出结果文件。

手工验证覆盖：

- 1000 样本检索。
- 10 万样本检索。
- 带 metadata 过滤的检索。

## 不做什么

第一版明确不做：

- 不调用 LLM。
- 不生成 embedding。
- 不引入 Elasticsearch、Milvus、FAISS 等检索服务。
- 不读取或修改原始 TSV。
- 不把 JSONL 输出文件提交到 Git。

