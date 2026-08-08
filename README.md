# 12345 工单实体抽取 v1

这是一次从零重写。旧版 SAG demo、v2–v8 语义契约、Oracle、标注工作台、旧数据库与
旧查询代码均不在本分支运行树中；历史仍保存在 Git commit 和归档 tag
`archive/qwen-semantic-v8-dev2-20260808`。

本仓库只交付代码、配置、合成测试和文档，不保存任何工单、模型、推理输出或人工标注。

## 服务器前提

服务器已具备以下资源，本项目不会下载、安装或删除它们：

```text
server-root/
├── data/
│   └── t_order_master.tsv
├── models/
│   └── Qwen3-4B/
└── app/                    # 本 Git 分支
```

还需要：

- 已创建且已安装 `vllm==0.8.5`、`transformers==4.51.3` 的 Conda 环境；
- 一张 NVIDIA V100；
- 服务器用户对 `RUN_DIR` 有写权限；
- 不要求代码联网。

## 唯一生产链路

```text
原始 TSV
→ 严格流式读取与坏行聚合
→ 四个语义字段及 metadata 本地 PII 脱敏
→ documents.private.jsonl（完整脱敏 RAG 事实源）
→ Qwen3-4B / vLLM
→ issues + 五类字符串数组
→ Python Unicode 精确 grounding
→ entities.private.jsonl（权威实体层）
→ entity_links.private.jsonl（确定性 issue 超边投影）
→ run.safe.json
```

模型只允许输出：

```json
{
  "issues": [
    {
      "objects": [],
      "problems": [],
      "questions": [],
      "locations": [],
      "requests": []
    }
  ]
}
```

每单一次 primary；只有 JSON 无效或 grounding 后整单为空时，最多一次 repair。字符串必须
逐字来自四个脱敏字段。Python 才生成 `field/start/end/evidence`，不做模糊匹配、同义改写、
canonical 或 confidence。

## 第一次服务器验证

激活服务器已有环境，在 `app/` 中执行：

```bash
conda activate <已有环境名>
bash scripts/verify_v1.sh
```

该命令只运行合成测试、Python 编译检查和 Shell 静态检查，不读取真实 TSV、不加载模型。

然后执行 16 条端到端 GPU smoke：

```bash
export DATA_PATH=/absolute/server-root/data/t_order_master.tsv
export MODEL_PATH=/absolute/server-root/models/Qwen3-4B
export RUN_DIR=/absolute/server-root/runs/entity-v1/smoke-001
export LIMIT=16
bash scripts/run_v1.sh
```

只回传：

- `prepare.safe.json`
- `diagnostics.safe.jsonl`
- `run.safe.json`
- 终端错误码（若失败）

不要回传 private JSONL、Prompt、模型原始响应或原始工单 ID。

## 恢复中断运行

不要重新 prepare。保持同一份代码、配置、documents 和模型：

```bash
unset LIMIT
export RESUME=1
export RUN_DIR=/absolute/server-root/runs/entity-v1/run-001
bash scripts/run_v1.sh
```

checkpoint 在模型调用前落盘，因此恢复不会重复已经开始的 primary 或 repair。若 repair 在
GPU 调用期间中断，该单以 `repair:interrupted` 私有 reject 结束，不会发起第三次生成。

## 全量运行

smoke、safe checker 和私有人工抽查通过后，使用一个新的空目录，且不要设置 `LIMIT`：

```bash
unset LIMIT RESUME
export RUN_DIR=/absolute/server-root/runs/entity-v1/full-001
bash scripts/run_v1.sh
```

`LIMIT` 只用于 smoke，不是正式可重现抽样。正式抽样以后应另建显式 identity manifest。

## 重要边界

- `documents.private.jsonl` 是完整脱敏 RAG 事实源，不能被 surface 实体图取代；
- v1 实体保持原文 surface，不做 alias/canonical，图节点可能碎片化；
- BM25/embedding 对完整脱敏文本的召回仍应保留；
- validator 通过率、accepted 率和 coverage 不代表实体准确率或 RAG/SAG 收益；
- 结构稳定后仍需私有人工实体/issue 审计和检索评测。

完整设计与产物契约见 [`docs/entity-extraction-v1.md`](docs/entity-extraction-v1.md)。
