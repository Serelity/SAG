# 01 - RAGFlow 项目速览

RAGFlow 是一个开源 RAG（Retrieval-Augmented Generation，检索增强生成）系统。它的核心目标是把文档解析、切分、向量化、检索、重排和 LLM 生成组织成一条可运行的知识库问答链路。

## 学习时先抓住这条主线

```text
文档上传
  -> 文档解析
  -> 文本切块
  -> Embedding 向量化
  -> 写入检索引擎
  -> 用户提问
  -> 检索相关片段
  -> LLM 生成答案
  -> 返回答案与引用
```

如果暂时没有可用 LLM API，也可以继续学习前半段：

- Docker 服务组成
- 系统配置
- 源码目录
- Web 与 API 的交互方式
- 数据库存储结构
- 排障方法

但完整 RAG 闭环最终需要两个模型能力：

- LLM：负责对话和答案生成。
- Embedding：负责把文档和问题转成向量。

## Docker 服务组成

当前 Docker 部署里主要有 5 个服务：

| 服务 | 作用 |
| --- | --- |
| `ragflow-cpu` | RAGFlow 主服务，包含 Web、API、任务处理等核心逻辑 |
| `mysql` | 保存用户、租户、知识库、文档、任务、模型配置等结构化数据 |
| `redis` / `valkey` | 缓存、任务状态、队列辅助 |
| `minio` | 对象存储，保存上传文件等二进制对象 |
| `es01` | Elasticsearch，保存文档切块、索引和向量检索数据 |

## 源码顶层目录

源码快照路径：

```text
G:\RAG\ragflow-main
```

重点目录：

| 目录 | 学习重点 |
| --- | --- |
| `docker/` | Docker Compose、环境变量、初始化脚本 |
| `api/` | 后端 Web API、数据库模型、服务层 |
| `rag/` | RAG 核心逻辑：检索、LLM、Prompt、文本处理 |
| `deepdoc/` | 文档解析、OCR、版面理解相关逻辑 |
| `web/` | 前端页面，React + Vite |
| `agent/` | Agent、插件、工具调用相关逻辑 |
| `sdk/` | 外部调用 RAGFlow 的 SDK |
| `conf/` | 默认配置、模型工厂、映射配置 |
| `test/` | 单元测试、接口测试、基准测试 |
| `cmd/` / `internal/` | Go 语言服务与内部模块 |

## 技术栈速览

| 层次 | 技术 |
| --- | --- |
| 部署 | Docker Compose |
| 后端主语言 | Python |
| 部分服务 | Go |
| 前端 | React、TypeScript、Vite |
| 数据库 | MySQL |
| 缓存/队列辅助 | Redis / Valkey |
| 对象存储 | MinIO |
| 检索引擎 | Elasticsearch / OpenSearch / Infinity 等 |
| 模型接入 | OpenAI、OpenAI-Compatible、Ollama、Jina、Cohere 等 |

## 第一阶段不要急着改源码

建议先做到：

- 能稳定启动和停止。
- 能看懂每个容器的作用。
- 能看懂 `.env` 与 `docker-compose.yml` 的关系。
- 能在日志里定位启动失败原因。
- 能说清楚没有 LLM API 时哪些功能不能完整验证。

这些是后续源码开发的基础。
