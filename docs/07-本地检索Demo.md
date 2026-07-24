# 07 - 本地检索 Demo

本节记录基于 12345 工单 JSONL 文档实现的第一版本地检索 demo。

这一步参考 RAGFlow 的核心思想：不要一开始就急着调用大模型，而是先把原始业务数据变成稳定、干净、可检索的知识单元，再验证检索链路是否有效。

## 本地检索的意义

本地检索是 RAG 的最低成本基线。

它的作用不是替代向量库，也不是替代 LLM，而是在接入复杂组件之前先回答三个问题：

- 每一行工单转成文本后，关键词还能不能搜到。
- 地区、分类、月份等 metadata 能不能作为过滤条件使用。
- 1000 行扩大到 10 万行后，数据格式、脱敏规则、检索程序是否还能稳定运行。

如果本地关键词检索都找不到合理工单，直接上 embedding 或向量数据库通常也不会稳定，因为问题很可能出在文档建模、字段选择或数据清洗阶段。

## 当前实现

第一版采用 Python 标准库实现，不依赖外部模型和数据库：

```text
脱敏 JSONL
  -> 加载文档
  -> 中文 bigram + 英文/数字 token 化
  -> 建立内存倒排索引
  -> BM25 风格打分
  -> metadata 过滤
  -> 输出 Top-K 结果
```

对应代码：

```text
src/ragflow_style_pipeline/text_tokenizer.py
src/ragflow_style_pipeline/local_search.py
src/ragflow_style_pipeline/search_jsonl.py
src/ragflow_style_pipeline/scan_jsonl_safety.py
```

## 运行命令

以下命令在 Windows PowerShell 中执行，但真正的 Python 程序运行在 Ubuntu/WSL 里。

### 1. 运行全部测试

```powershell
wsl -d RAGFlow-Ubuntu -- bash -lc "cd /mnt/g/RAG/ragflow-learning-plan && PYTHONPATH=src python3 -m unittest discover -s tests -t ."
```

含义：

- `wsl -d RAGFlow-Ubuntu`：进入名为 `RAGFlow-Ubuntu` 的 Ubuntu 子系统。
- `bash -lc "..."`：让 Ubuntu 用 bash 执行引号里的命令。
- `cd /mnt/g/RAG/ragflow-learning-plan`：进入学习仓库目录。
- `PYTHONPATH=src`：告诉 Python 从 `src` 目录寻找项目代码。
- `python3 -m unittest discover -s tests -t .`：运行 `tests` 目录下的所有单元测试。

### 2. 重新生成 1000 样本

```powershell
wsl -d RAGFlow-Ubuntu -- bash -lc "cd /mnt/g/RAG/ragflow-learning-plan && PYTHONPATH=src python3 -m ragflow_style_pipeline.export_jsonl --input /mnt/g/12345_pro_promax/data/t_order_master.tsv --output outputs/t_order_master.sample.jsonl --quality-report outputs/t_order_master.sample.quality.json --limit 1000"
```

含义：

- `python3 -m ragflow_style_pipeline.export_jsonl`：运行 TSV 转 JSONL 的导出程序。
- `--input ...t_order_master.tsv`：指定原始 TSV 文件位置，只读取，不修改。
- `--output outputs/t_order_master.sample.jsonl`：输出脱敏后的 JSONL 样本。
- `--quality-report outputs/t_order_master.sample.quality.json`：输出质量报告。
- `--limit 1000`：只读取前 1000 行，适合快速验证。

### 3. 重新生成 10 万样本

```powershell
wsl -d RAGFlow-Ubuntu -- bash -lc "cd /mnt/g/RAG/ragflow-learning-plan && PYTHONPATH=src python3 -m ragflow_style_pipeline.export_jsonl --input /mnt/g/12345_pro_promax/data/t_order_master.tsv --output outputs/t_order_master.100k.jsonl --quality-report outputs/t_order_master.100k.quality.json --limit 100000"
```

含义：

- 和 1000 样本命令相同。
- `--limit 100000`：扩大到前 10 万行，用来暴露小样本看不到的数据质量问题。

### 4. 扫描脱敏风险

```powershell
wsl -d RAGFlow-Ubuntu -- bash -lc "cd /mnt/g/RAG/ragflow-learning-plan && PYTHONPATH=src python3 -m ragflow_style_pipeline.scan_jsonl_safety --input outputs/t_order_master.100k.jsonl --output outputs/t_order_master.100k.safety.json"
```

含义：

- `scan_jsonl_safety`：扫描 JSONL 中疑似未脱敏的手机号、身份证号、显式姓名标签。
- `--input outputs/t_order_master.100k.jsonl`：扫描 10 万样本 JSONL。
- `--output outputs/t_order_master.100k.safety.json`：保存安全扫描报告。
- 这个工具只输出统计数字，不打印工单原文。

### 5. 1000 样本检索

```powershell
wsl -d RAGFlow-Ubuntu -- bash -lc "cd /mnt/g/RAG/ragflow-learning-plan && PYTHONPATH=src python3 -m ragflow_style_pipeline.search_jsonl --input outputs/t_order_master.sample.jsonl --query '武进区 夜间 摆摊 扰民' --top-k 5 --output outputs/search_demo.sample.json"
```

含义：

- `search_jsonl`：运行本地检索程序。
- `--input outputs/t_order_master.sample.jsonl`：从 1000 样本中检索。
- `--query '武进区 夜间 摆摊 扰民'`：查询语句。
- `--top-k 5`：返回前 5 条结果。
- `--output outputs/search_demo.sample.json`：保存安全结果，只保存 snippet，不保存完整正文。

### 6. 10 万样本检索

```powershell
wsl -d RAGFlow-Ubuntu -- bash -lc "cd /mnt/g/RAG/ragflow-learning-plan && PYTHONPATH=src python3 -m ragflow_style_pipeline.search_jsonl --input outputs/t_order_master.100k.jsonl --query '拖欠工资 工地 工资 未发' --top-k 5 --output outputs/search_demo.100k.salary.json"
```

含义：

- `--input outputs/t_order_master.100k.jsonl`：从 10 万样本中检索。
- `--query '拖欠工资 工地 工资 未发'`：查询工资拖欠类工单。
- `--top-k 5`：返回前 5 条结果。

### 7. 10 万样本 + 地区过滤

```powershell
wsl -d RAGFlow-Ubuntu -- bash -lc "cd /mnt/g/RAG/ragflow-learning-plan && PYTHONPATH=src python3 -m ragflow_style_pipeline.search_jsonl --input outputs/t_order_master.100k.jsonl --query '占道经营 摆摊' --area '武进区' --top-k 5 --output outputs/search_demo.100k.wujin_stall.json"
```

含义：

- `--query '占道经营 摆摊'`：查询摆摊占道类工单。
- `--area '武进区'`：只在 `metadata.area_code_area = 武进区` 的工单中检索。
- 这一步验证 metadata 过滤的价值。

## 当前验证结果

1000 样本导出：

```json
{
  "rows_read": 1000,
  "documents_written": 995,
  "rows_skipped_bad_field_count": 5,
  "rows_skipped_polluted_text": 0,
  "redactions": {
    "phone": 19,
    "alnum_id": 1096,
    "id_card": 22,
    "long_number": 2,
    "numeric_id": 1,
    "name": 9
  }
}
```

10 万样本导出：

```json
{
  "rows_read": 100000,
  "documents_written": 99133,
  "rows_skipped_bad_field_count": 739,
  "rows_skipped_polluted_text": 128,
  "redactions": {
    "phone": 2839,
    "alnum_id": 108977,
    "id_card": 2350,
    "long_number": 615,
    "numeric_id": 294,
    "name": 1217
  }
}
```

10 万样本安全扫描：

```json
{
  "documents_scanned": 99133,
  "possible_unredacted_phone": 0,
  "possible_unredacted_id_card": 0,
  "possible_unredacted_name_label": 0
}
```

检索效果：

- `武进区 夜间 摆摊 扰民` 在 1000 样本中能召回占道经营、夜间扰民、市容管理相关工单。
- `拖欠工资 工地 工资 未发` 在 10 万样本中能召回民生保障 / 劳动纠纷 / 拖欠薪资相关工单。
- `占道经营 摆摊 --area 武进区` 能把结果限制在武进区。

## 当前局限

第一版仍有局限：

- 中文分词使用 bigram，不理解真正的词语边界。
- BM25 是关键词匹配，不理解同义表达。
- 对没有明确标签的自由文本姓名，规则脱敏无法保证完全覆盖。
- 每次查询都会重新加载 JSONL 并建索引，适合 demo，不适合长期服务。

后续成熟 RAG 方案应该继续做：

- 引入更可靠的中文分词或 Elasticsearch。
- 使用 embedding 做语义召回。
- 使用 metadata 做精确过滤。
- 使用 reranker 对候选结果重排。
- 对脱敏加入 NER、抽样人工审核和安全扫描闭环。

