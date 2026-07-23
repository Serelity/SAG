# RAGFlow Learning Repository Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a clean Chinese learning-record repository for a 12-week RAGFlow study plan.

**Architecture:** The repository stores documentation, lab notes, and troubleshooting records only. It references the upstream RAGFlow project without copying the full source tree or local runtime data.

**Tech Stack:** Markdown, Git, GitHub.

---

### Task 1: Repository safety files

**Files:**

- Create: `.gitignore`
- Create: `LICENSE`

- [x] **Step 1: Add `.gitignore`**

Protect local secrets, Docker runtime data, editor files, logs, and caches from being committed.

- [x] **Step 2: Add `LICENSE`**

Use MIT license for the user's learning notes.

### Task 2: Main README

**Files:**

- Modify: `README.md`

- [x] **Step 1: Rewrite README in Chinese**

Include goal, current status, repository layout, local deployment commands, and learning principles.

### Task 3: Documentation pages

**Files:**

- Create: `docs/00-环境与部署记录.md`
- Create: `docs/01-RAGFlow项目速览.md`
- Create: `docs/02-12周学习计划.md`
- Create: `docs/03-模型配置说明.md`
- Create: `docs/04-排障记录.md`
- Create: `docs/05-源码阅读路线.md`

- [x] **Step 1: Record environment and deployment**

Capture WSL, Docker Desktop, RAGFlow version, deployment directory, ports, start/stop commands, and verification commands.

- [x] **Step 2: Summarize RAGFlow project**

Document the RAG pipeline, Docker services, top-level source directories, and technology stack.

- [x] **Step 3: Write 12-week learning plan**

Break the study into weekly goals, tasks, and outputs.

- [x] **Step 4: Record model configuration issue**

Explain why RAGFlow needs both Chat Completions and Embeddings, and why a Responses-only API is insufficient for current OpenAI-Compatible configuration.

- [x] **Step 5: Record troubleshooting**

Capture Docker WSL integration, port conflict, image pull, entrypoint, SubAPI, and reboot recovery cases.

- [x] **Step 6: Write source reading route**

Provide ordered reading paths for Docker, model providers, knowledge base, document parsing, retrieval, and frontend.

### Task 4: Lab and notes structure

**Files:**

- Create: `labs/README.md`
- Create: `labs/week-01.md`
- Create: `notes/README.md`

- [x] **Step 1: Add lab template**

Define how experiments should be recorded.

- [x] **Step 2: Add week 1 record**

Summarize completed deployment work and next week tasks.

- [x] **Step 3: Add notes index**

Define naming convention for future notes.

### Task 5: Verification and Git handoff

**Files:**

- No new files.

- [ ] **Step 1: Inspect status**

Run:

```powershell
git status --short
```

Expected: only intended docs and repository metadata are listed.

- [ ] **Step 2: Commit**

Run:

```powershell
git add README.md .gitignore LICENSE docs labs notes
git commit -m "docs: add ragflow learning plan"
```

Expected: commit succeeds.

- [ ] **Step 3: Prepare GitHub upload**

If no `origin` remote exists, ask user for a repository URL or ask them to create one on GitHub.
