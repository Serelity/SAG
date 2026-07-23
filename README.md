# RAGFlow 12 周学习计划

这是我的 RAGFlow 学习记录仓库。目标不是复制上游源码，而是把部署、排障、源码阅读、实验过程和阶段复盘整理成一个可复现的学习项目。

> 上游项目：[`infiniflow/ragflow`](https://github.com/infiniflow/ragflow)  
> 当前学习版本：`v0.26.4`

## 学习目标

- 在 Windows + WSL2 + Docker Desktop 环境稳定部署 RAGFlow。
- 理解 RAGFlow 的 Docker 服务组成和启动链路。
- 理解 RAGFlow 的核心源码结构：Web、API、RAG、DeepDoc、Agent、SDK。
- 通过 12 周、每周 15 小时以上的节奏，形成持续可复盘的学习记录。
- 最终把学习文档、实验记录和问题清单上传到自己的 GitHub。

## 当前状态

- [x] WSL2 安装完成。
- [x] Ubuntu 24.04 LTS 发行版创建完成。
- [x] Docker Desktop 与 WSL2 集成修复完成。
- [x] RAGFlow `v0.26.4` CPU Docker 部署完成。
- [x] Web 页面可访问：`http://localhost`
- [x] API 端口可访问：`localhost:9380`
- [x] 已确认本地 MySQL `3306` 冲突，并将 RAGFlow MySQL 暴露端口改为 `13306`。
- [x] 已记录 `SubAPI / Responses API` 与 RAGFlow 当前 OpenAI-Compatible 接口的适配问题。
- [ ] 配置一个同时支持 Chat Completions 与 Embeddings 的模型供应商。
- [ ] 创建第一个数据集并完成一次文档解析实验。
- [ ] 开始系统源码阅读。

## 仓库内容

```text
.
├── README.md
├── docs/
│   ├── 00-环境与部署记录.md
│   ├── 01-RAGFlow项目速览.md
│   ├── 02-12周学习计划.md
│   ├── 03-模型配置说明.md
│   ├── 04-排障记录.md
│   └── 05-源码阅读路线.md
├── labs/
│   ├── README.md
│   └── week-01.md
└── notes/
    └── README.md
```

## 本地部署摘要

我的本地部署目录位于 WSL 内：

```bash
~/deploy/ragflow-v0.26.4
```

关键启动命令：

```powershell
wsl -d RAGFlow-Ubuntu -- bash -lc "cd ~/deploy/ragflow-v0.26.4 && docker compose -f docker-compose.yml start"
```

关键状态检查命令：

```powershell
wsl -d RAGFlow-Ubuntu -- bash -lc "cd ~/deploy/ragflow-v0.26.4 && docker compose -f docker-compose.yml ps"
```

安全停止但保留数据：

```powershell
wsl -d RAGFlow-Ubuntu -- bash -lc "cd ~/deploy/ragflow-v0.26.4 && docker compose -f docker-compose.yml stop"
```

注意：不要使用带 `-v` 的 `docker compose down`，否则会删除数据卷。

## 学习原则

- 每次学习都留下记录：目标、操作、现象、结论。
- 遇到问题先定位根因，不随机改配置。
- 不把 API Key、密码、Cookie、Token 写进仓库。
- 不直接修改上游源码快照，实验修改应单独记录或另开分支。
- 先跑通 Docker 部署，再进入源码开发。
