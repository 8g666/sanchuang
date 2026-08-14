# DocGuard-RAG

DocGuard-RAG 是一个通用文档 RAG、文本纠错与质量审查包。它既可以通过 `DocGuard` 直接嵌入 Python 程序，也可以启动 REST 服务，再通过 `DocGuardClient` 或 HTTP 接口调用。

当前版本：`0.6.0`。

## 主要能力

- 解析 TXT、Markdown、HTML、DOCX、PDF
- 文档分块、embedding、混合检索与来源引用
- 问题纠错、查询改写、相关性评分、证据充分性判断
- 答案验证、有限次数重试和证据不足拒答
- 文本纠错与文档质量审查
- 文档查询、替换、重建索引和完整删除
- 本地向量库或 Qdrant
- 进程内线程任务或 Celery/Redis 任务队列
- 多租户隔离、Python SDK 和 REST API

## 选择运行方式

| 使用场景 | 推荐方式 | 外部服务 |
| --- | --- | --- |
| 本地开发、单元测试、嵌入现有 Python 程序 | `DocGuard` | 不需要 |
| 单机 HTTP 服务 | REST API + 本地向量库 | 不需要 |
| 多进程或生产部署 | REST API + Qdrant + Redis/Celery | Qdrant、Redis，建议 PostgreSQL |

如果只是想快速体验，从“本地最小示例”开始即可，不需要 Docker。

## 安装

要求 Python 3.11 或更高版本。

### Windows 开发环境

```powershell
Set-Location D:\CQCAI\CIGAI\docguard-rag
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

激活成功后，命令行前面会出现 `(.venv)`。退出虚拟环境：

```powershell
deactivate
```

### 安装构建产物

```powershell
python -m pip install .\dist\docguard_rag-0.6.0-py3-none-any.whl
```

需要 Celery/Redis 时安装分布式依赖：

```powershell
python -m pip install -e ".[distributed]"
```

## 本地最小示例

本地模式使用 SQLite、本地文件存储、内置 embedding、抽取式回答和进程内向量库，不需要启动 REST、Qdrant 或 Redis。

```python
from docguard_rag import DocGuard

with DocGuard(
    database_url="sqlite:///./example.db",
    upload_dir="./example-uploads",
) as guard:
    document = guard.documents.upload("example.md")

    result = guard.query(
        "文档的主要结论是什么？",
        document_ids=[document.document_id],
    )

    print("状态：", result.status)
    print("回答：", result.answer)
    for source in result.sources:
        print(source.document_id, source.page, source.text)
```

建议使用 `with DocGuard(...)`，退出代码块时会自动关闭线程池和外部连接。

## 文档操作

### 上传

```python
document = guard.documents.upload(
    "example.pdf",
    collection_id="manuals",
    enable_review=False,
    metadata={"department": "研发部", "revision": 1},
)

print(document.document_id)
print(document.status)        # READY
print(document.blocks_count)
print(document.chunks_count)
```

异步上传：

```python
import time

document = guard.documents.upload("large.pdf", async_mode=True)

while True:
    job = guard.jobs.get(document.job_id)
    print(job.status, job.progress, job.current_stage)
    if job.status in {"COMPLETED", "FAILED"}:
        break
    time.sleep(1)
```

本地异步模式使用进程内线程池；REST 服务配置 Celery 后使用 Redis 队列。

### 查询文档状态、块和分块

```python
status = guard.documents.get(document.document_id)
blocks = guard.documents.blocks(document.document_id)
chunks = guard.documents.chunks(document.document_id)

print(status.filename, status.chunk_count)
print(blocks[0].text)
print(chunks[0].text)
```

### 替换文档

替换会保留原来的 `document_id`，重新解析、分块和建立索引：

```python
updated = guard.documents.update(
    document.document_id,
    "replacement.pdf",
    metadata={"revision": 2},
)
```

### 重建索引

使用已保存的原始文件重新解析并生成索引：

```python
rebuilt = guard.documents.reindex(document.document_id)
```

### 删除

删除数据库记录、文档块、分块、向量、审查问题和保存的原始文件：

```python
deleted = guard.documents.delete(document.document_id)
print(deleted.status)  # DELETED
```

## Corrective RAG 查询

`query()` 默认运行纠错检索闭环：

```text
原始问题
  → 错别字修正和问题改写
  → 混合检索与重排
  → 检索相关性评分
  → 证据充分性判断
  → 答案生成
  → 引用与事实一致性验证
  → 重试或拒答
```

```python
result = guard.query(
    "文挡中接口应该怎么掉用？",
    document_ids=[document.document_id],
    collection_id=None,
    top_k=5,
    corrective=True,
    max_retries=2,
    include_sources=True,
)

print(result.status)
print(result.original_query)
print(result.effective_query)
print(result.query_rewritten)
print(result.verified)
print(result.attempts)
```

常见状态：

- `ANSWERED`：证据充分且回答通过验证
- `INSUFFICIENT_EVIDENCE`：文档证据不足，系统拒绝猜测
- `ANSWER_UNVERIFIED`：生成了候选答案，但未通过一致性验证

如需兼容旧版单轮检索行为：

```python
result = guard.query("问题", corrective=False)
```

## 文本纠错

文本纠错不要求先上传文档：

```python
result = guard.correct(
    "系统能够被直接掉用。。",
    check_types=["TYPO", "GRAMMAR", "SEMANTIC_UNCLEAR"],
    mode="BALANCED",
    apply_changes=True,
)

print(result.original_text)
print(result.corrected_text)
print(result.changed)
print(result.issues)
```

## 文档质量审查

```python
review = guard.reviews.create(
    document_ids=[document.document_id],
    check_types=["TYPO", "SEMANTIC_UNCLEAR"],
    mode="BALANCED",
)

issues = guard.reviews.list_issues(
    review.review_id,
    min_confidence=0.75,
)

for issue in issues.items:
    print(issue.issue_type, issue.problem_text, issue.suggestion)
```

更新问题处理状态：

```python
guard.issues.update(
    issue.issue_id,
    status="ACCEPTED",
    comment="已确认并准备修改",
)
```

## 启动 REST 服务

```powershell
Set-Location D:\CQCAI\CIGAI\docguard-rag
.\.venv\Scripts\Activate.ps1
python -m uvicorn docguard_rag.asgi:app --host 127.0.0.1 --port 8000
```

健康检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/ready
```

也可以使用命令行入口：

```powershell
docguard-rag --host 0.0.0.0 --port 8000
```

## HTTP SDK

`DocGuardClient` 的资源命名和 `DocGuard` 基本一致：

```python
import time

from docguard_rag import DocGuardClient

with DocGuardClient(
    base_url="http://127.0.0.1:8000",
    api_key=None,
) as client:
    document = client.documents.upload("example.pdf", async_mode=True)

    while True:
        job = client.jobs.get(document.job_id)
        if job.status in {"COMPLETED", "FAILED"}:
            break
        time.sleep(1)

    result = client.query(
        "主要结论是什么？",
        document_ids=[document.document_id],
    )

    client.documents.update(document.document_id, "replacement.pdf")
    client.documents.reindex(document.document_id)
    client.documents.delete(document.document_id)
```

## 直接调用 REST API

PowerShell 上传示例：

```powershell
$uploadJson = curl.exe -s `
  -X POST "http://127.0.0.1:8000/v1/documents" `
  -F "file=@example.txt;type=text/plain" `
  -F "async=true"

$upload = $uploadJson | ConvertFrom-Json
$documentId = $upload.data.document_id
$jobId = $upload.data.job_id
```

查询任务：

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/v1/jobs/$jobId"
```

查询文档：

```powershell
$body = @{
    query = "文档的主要内容是什么？"
    document_ids = @($documentId)
    top_k = 5
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/v1/query" `
  -ContentType "application/json" `
  -Body $body
```

### REST 接口列表

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| POST | `/v1/documents` | 上传文档 |
| GET | `/v1/documents/{document_id}` | 查询文档状态 |
| PUT | `/v1/documents/{document_id}` | 替换文档 |
| DELETE | `/v1/documents/{document_id}` | 删除文档 |
| POST | `/v1/documents/{document_id}/reindex` | 重建索引 |
| GET | `/v1/documents/{document_id}/blocks` | 获取文档块 |
| GET | `/v1/documents/{document_id}/chunks` | 获取分块 |
| POST | `/v1/query` | RAG 查询 |
| POST | `/v1/correct` | 文本纠错 |
| POST | `/v1/reviews` | 创建审查 |
| GET | `/v1/reviews/{review_id}/issues` | 查询审查问题 |
| PATCH | `/v1/issues/{issue_id}` | 更新问题状态 |
| GET | `/v1/jobs/{job_id}` | 查询异步任务 |
| GET | `/health` | 存活检查 |
| GET | `/ready` | 就绪检查 |

## 使用 Qdrant

默认向量后端是 `local`。本地模式不需要 Qdrant，但每次检索会从 SQL 数据库加载规范向量，适合开发和小规模数据。

### 使用 Docker 启动 Qdrant

```powershell
docker run -d --name docguard-qdrant `
  -p 6333:6333 `
  -v docguard_qdrant_data:/qdrant/storage `
  qdrant/qdrant:latest
```

检查状态：

```powershell
Invoke-RestMethod http://127.0.0.1:6333/readyz
```

预期返回：

```text
all shards are ready
```

Qdrant 控制台：<http://127.0.0.1:6333/dashboard>

### 配置 DocGuard

在 API 和 worker 的环境中设置：

```dotenv
DOCGUARD_VECTOR_STORE_PROVIDER=qdrant
DOCGUARD_QDRANT_URL=http://127.0.0.1:6333
DOCGUARD_QDRANT_API_KEY=
DOCGUARD_QDRANT_COLLECTION=docguard_chunks
DOCGUARD_QDRANT_TIMEOUT=30
DOCGUARD_EMBEDDING_DIMENSION=128
NO_PROXY=127.0.0.1,localhost
```

集合不存在时会自动创建。已有集合的向量维度必须与 embedding provider 的输出维度一致。SQL 数据库仍保存规范向量记录，便于审计和恢复。

如果访问本地 Qdrant 得到 `502 Bad Gateway`，通常是系统代理截获了本地请求。PowerShell 中设置：

```powershell
$env:NO_PROXY = "127.0.0.1,localhost,$($env:NO_PROXY)"
```

## 使用 Redis/Celery

### 启动 Redis

```powershell
docker run -d --name docguard-redis `
  -p 6379:6379 `
  -v docguard_redis_data:/data `
  redis:7-alpine redis-server --appendonly yes
```

验证：

```powershell
docker exec docguard-redis redis-cli ping
```

预期返回 `PONG`。

### 配置任务队列

```dotenv
DOCGUARD_TASK_BACKEND=celery
DOCGUARD_CELERY_BROKER_URL=redis://127.0.0.1:6379/0
DOCGUARD_CELERY_RESULT_BACKEND=redis://127.0.0.1:6379/1
DOCGUARD_CELERY_TASK_MAX_RETRIES=3
DOCGUARD_CELERY_TASK_RETRY_DELAY=5
```

API 服务与 worker 必须：

- 使用相同的 `DOCGUARD_DATABASE_URL`
- 使用相同且可共享的 `DOCGUARD_UPLOAD_DIR`
- 使用相同的 Qdrant collection 和模型配置
- 能同时访问 Redis 和 Qdrant

### Windows 开发 worker

Celery 不正式支持 Windows 生产部署，本地验证使用 `solo` pool：

```powershell
python -m celery `
  -A docguard_rag.server.services.task_dispatcher:celery_app worker `
  --pool=solo `
  --hostname=worker1@%h `
  --concurrency=1 `
  --prefetch-multiplier=1 `
  --loglevel=INFO
```

测试第二个 worker 时，在另一个 PowerShell 窗口中使用不同名称：

```powershell
python -m celery `
  -A docguard_rag.server.services.task_dispatcher:celery_app worker `
  --pool=solo `
  --hostname=worker2@%h `
  --concurrency=1 `
  --prefetch-multiplier=1 `
  --loglevel=INFO
```

### Linux 生产 worker

```bash
celery \
  -A docguard_rag.server.services.task_dispatcher:celery_app worker \
  --concurrency=4 \
  --prefetch-multiplier=1 \
  --loglevel=INFO
```

Celery 任务使用 JSON 序列化、延迟确认、worker 丢失时重新入队和有限次数重试。

注意：当前版本在 Celery 等待下一次重试时，`GET /v1/jobs/{job_id}` 可能短暂显示 `FAILED`，随后再次变为 `PROCESSING` 或 `COMPLETED`。故障恢复测试中应在整个重试预算结束后再判断最终失败。

## 推荐的生产配置

- API 和 worker 部署在 Linux 容器或 Kubernetes
- 使用 PostgreSQL 替代 SQLite，避免多 worker 写锁竞争
- 使用持久化 Redis 或高可用 Redis/Sentinel
- 使用持久化或集群化 Qdrant
- API 和 worker 挂载同一共享上传目录
- 为 Qdrant、Redis、数据库和模型服务配置健康检查
- 为 Celery 配置监控、任务超时和失败告警
- 不要把 API Key、数据库密码或模型密钥提交到版本库

## 模型 Provider

默认 provider 可以离线运行：

- embedding：确定性 hash embedding
- reranker：词项重叠评分
- answer：抽取式回答
- reviewer：本地错别字和语义检查

切换到兼容 API：

```dotenv
DOCGUARD_EMBEDDING_PROVIDER=compatible_api
DOCGUARD_EMBEDDING_BASE_URL=https://model.example.com/v1
DOCGUARD_EMBEDDING_API_KEY=secret
DOCGUARD_EMBEDDING_MODEL_NAME=embedding-model

DOCGUARD_RERANKER_PROVIDER=compatible_api
DOCGUARD_RERANKER_BASE_URL=https://rerank.example.com
DOCGUARD_RERANKER_API_KEY=secret
DOCGUARD_RERANKER_MODEL_NAME=reranker-model

DOCGUARD_LLM_PROVIDER=compatible_api
DOCGUARD_LLM_BASE_URL=https://model.example.com/v1
DOCGUARD_LLM_API_KEY=secret
DOCGUARD_LLM_MODEL_NAME=chat-model
```

Embedding 和 LLM provider 使用 OpenAI-compatible 请求格式；reranker 使用常见的 `query + documents + top_n` 请求结构。

## 多租户认证

```dotenv
DOCGUARD_AUTH_ENABLED=true
DOCGUARD_API_KEYS_JSON={"key-a":{"tenant_id":"tenant-a","owner_id":"user-a"},"key-b":"tenant-b"}
```

客户端：

```python
client = DocGuardClient(
    "http://127.0.0.1:8000",
    api_key="key-a",
)
```

启用认证后，文档、检索、审查、问题和任务接口都会执行租户过滤。

## 数据库迁移

生产部署应在启动 API 和 worker 前运行迁移：

```powershell
docguard-rag --migrate
```

或：

```powershell
alembic upgrade head
```

## 测试

运行全部自动化测试：

```powershell
Set-Location D:\CQCAI\CIGAI\docguard-rag
.\.venv\Scripts\Activate.ps1
pytest -q
```

代码规范检查：

```powershell
python -m ruff check docguard_rag tests
python -m ruff format --check docguard_rag tests
```

真实分布式验收建议覆盖：

1. 异步上传最终为 `COMPLETED`
2. Qdrant 中可以检索到文档 points
3. 替换和重建索引后不再返回旧内容
4. 删除后 SQL 记录、Qdrant points 和上传文件均清空
5. 两个 worker 可以共同消费任务
6. Qdrant 暂停后任务能够重试并恢复
7. Redis 暂停时 API 返回 503，恢复后新任务正常执行

## 常见问题

### PowerShell 无法运行 Activate.ps1

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### `docker` 不是可识别的命令

安装并启动 Docker Desktop，重新打开 PowerShell，再执行：

```powershell
docker version
```

输出必须同时包含 `Client` 和 `Server`。

### Docker CLI 无法连接 daemon

启动 Docker Desktop，等待左下角显示 `Engine running`，再执行 Docker 命令。

### 在 `docker-desktop:~#` 中无法运行命令

`docker-desktop` 是 Docker Desktop 内部 WSL 发行版。执行 `exit` 返回 Windows PowerShell；`docker` 和 `Invoke-RestMethod` 应在 PowerShell 中运行。

### pytest 报临时目录 PermissionError

项目会为每次测试生成独立临时目录并在结束后清理。确保使用项目最新的 `tests/conftest.py`，然后运行：

```powershell
pytest -q
```

### 异步任务一直停留在 PENDING

检查：

```powershell
docker exec docguard-redis redis-cli ping
```

并确认 Celery worker 窗口已经显示 `ready`，API 和 worker 使用相同环境变量。

### Qdrant 返回 502

确认 Qdrant 健康：

```powershell
Invoke-RestMethod http://127.0.0.1:6333/readyz
$env:NO_PROXY = "127.0.0.1,localhost,$($env:NO_PROXY)"
```

## 可替换组件接口

```python
from docguard_rag.interfaces import (
    Chunker,
    DocumentParser,
    DocumentReviewer,
    EmbeddingProvider,
    LLMProvider,
    Reranker,
    Retriever,
    VectorStore,
)
```

完整环境变量示例见 `.env.example`。
