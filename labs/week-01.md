# Week 01 - 环境部署实验

## 本周目标

- 完成 Windows + WSL2 + Docker Desktop 环境。
- 完成 RAGFlow `v0.26.4` CPU 模式部署。
- 掌握启动、停止、查看状态和查看日志的基本命令。

## 已完成

- WSL2 安装完成。
- `RAGFlow-Ubuntu` Ubuntu 24.04 LTS 安装完成。
- WSL 资源配置完成：16GB 内存、10 核 CPU、8GB Swap。
- Docker Desktop 数据目录迁移到 `D:\DockerData\DockerDesktopWSL`。
- Docker Desktop 与 WSL 集成问题已修复。
- RAGFlow Docker 镜像已拉取。
- RAGFlow 容器已启动。
- Web/API 可访问。

## 常用命令

启动：

```powershell
wsl -d RAGFlow-Ubuntu -- bash -lc "cd ~/deploy/ragflow-v0.26.4 && docker compose -f docker-compose.yml start"
```

查看状态：

```powershell
wsl -d RAGFlow-Ubuntu -- bash -lc "cd ~/deploy/ragflow-v0.26.4 && docker compose -f docker-compose.yml ps"
```

查看主服务日志：

```powershell
wsl -d RAGFlow-Ubuntu -- bash -lc "cd ~/deploy/ragflow-v0.26.4 && docker compose -f docker-compose.yml logs --tail=120 ragflow-cpu"
```

停止但保留数据：

```powershell
wsl -d RAGFlow-Ubuntu -- bash -lc "cd ~/deploy/ragflow-v0.26.4 && docker compose -f docker-compose.yml stop"
```

## 本周问题

### Docker Desktop WSL 集成异常

现象：Docker Desktop 集成文件在 Ubuntu 内没有正确挂载。

结论：完整冷重启后恢复。

### MySQL 端口冲突

现象：Windows 本机 `MySQL80` 占用 `3306`。

处理：RAGFlow 宿主机暴露端口改为 `13306`，容器内部保持 `3306`。

### ragflow-cpu 启动重启

现象：入口脚本找不到 `tools/scripts/run_migrations.sh`。

处理：部署副本中注释 `./entrypoint.sh:/ragflow/entrypoint.sh` 挂载，使用镜像内置入口脚本。

## 下周任务

- 阅读 `docker-compose.yml`。
- 阅读 `docker-compose-base.yml`。
- 梳理 5 个核心服务的职责。
- 画出服务依赖关系。
