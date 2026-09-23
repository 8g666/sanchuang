# DocGuard-RAG

DocGuard-RAG 是一个通用文档 RAG、文本纠错与质量审查包。它既可以通过 `DocGuard` 直接嵌入 Python 程序，也可以启动 REST 服务，再通过 `DocGuardClient` 或 HTTP 接口调用。

当前版本：`0.6.0`。

## 主要能力

- 解析 TXT、Markdown、HTML、DOCX、PPTX、PDF，以及 PNG、JPG、JPEG、TIFF、BMP、WEBP 图片 OCR
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

### 安装本地包

```powershell
python -m pip install .
```

需要 Celery/Redis 时安装分布式依赖：

```powershell
python -m pip install -e ".[distributed]"
```

需要图片 OCR 时安装图片依赖，并在操作系统中安装 Tesseract OCR：

```powershell
python -m pip install -e ".[image]"
```

Windows 安装完成后请确认 `tesseract.exe` 已加入 PATH：

```powershell
tesseract --version
```

Linux 可通过系统包管理器安装 Tesseract。若命令不可用，Python 依赖虽然安装成功，但图片上传仍会提示 OCR 引擎不可用。

图片上传后会先通过 Tesseract 提取文字，再进入与 DOCX、Markdown、PDF 相同的分块、向量化和检索流程。中文图片建议安装 `chi_sim` 语言包，并配置：

```dotenv
DOCGUARD_OCR_LANG=chi_sim+eng
DOCGUARD_OCR_TESSERACT_CMD=
```

如果 `tesseract.exe` 不在 PATH，可直接填写完整路径，例如
`C:\\Program Files\\Tesseract-OCR\\tesseract.exe`。

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

### 图片数据源

图片可以直接按普通文档上传：

```python
document = guard.documents.upload("申报指南截图.png")
result = guard.query(
    "截图中的申报截止时间是什么？",
    document_ids=[document.document_id],
)
```

构建混合类型知识库时，可以一次提交多个 TXT、Markdown、DOCX、PPTX、PDF、HTML 和图片文件：

```python
documents = guard.documents.upload_many(
    ["指南.docx", "申报说明.pdf", "介绍.pptx", "流程.md", "流程截图.png"],
    collection_id="tongjing-san-chuang",
    metadata={"source": "official-materials"},
)
```

如果图片没有可识别文字，上传会返回“没有找到可读文本”；如果未安装 OCR 依赖，错误信息会提示安装 `docguard-rag[image]` 和 Tesseract。

### PPTX 数据源

PPTX 和 DOCX 一样只依赖标准库，安装后即可上传，无需 `python-pptx`。

| 幻灯片内容 | 解析结果 |
| --- | --- |
| 标题占位符（`title`、`ctrTitle`） | `heading`，同时作为该页所有块的 `section_path` |
| 正文、表格单元格、组合形状内的文字 | `paragraph` |
| 演讲者备注 | `paragraph`，页码归属其所在的幻灯片 |
| 母版与版式（`slideMaster`、`slideLayout`） | 不读取，避免模板占位文字在每页重复 |
| 页码、日期、页脚字段（`a:fld`） | 丢弃，它们是模板缓存值而非作者内容 |

`page_number` 就是幻灯片序号，因此检索结果的引用能直接落到「第几页」。没有标题占位符的页用「幻灯片 N」兜底，配合分块策略里的页边界，保证一页不会被并进上一页的分块里。

实测两份真实演示文稿：

| 文件 | 页数 | 块数 | 可索引字数 | 分块数 |
| --- | --- | --- | --- | --- |
| 用户操作手册 | 31 | 302 | 2535 | 31 |
| OPC 三创平台建设方案 V13 | 51 | 2009 | 22999 | 73 |

第二份文件导出的 PDF 解析结果是 **0 字**（导出成了纯图片），而 pptx 原件能拿到近 2.3 万字。**同一份材料既有 pptx 又有 PDF 时，优先用 pptx。**

两点限制：

- `DOCGUARD_MAX_UPLOAD_SIZE` 默认 20 MB，插图较多的演示文稿很容易超过（实测有 102 MB、188 MB 的），上传前需要调大这个值。
- 解析器只读文字层，**不做幻灯片内嵌图片的 OCR**。内容全靠截图承载的手册（例如上面那份 31 页的操作手册内嵌了 38 张截图）只能拿到标题和图注，图里的操作步骤提不出来，需要对 `ppt/media/*` 另做 OCR。

### 分块策略

解析器按结构切块：Markdown 和 TXT 一行一块，DOCX 一段一块，PDF 一行一块，PPTX 一个段落一块。真实文档里这些块普遍很短——幻灯片导出的 PDF 平均每块只有 11 字，逐行解析的 Markdown 约 27 字。如果一块直接产出一个分块，`DOCGUARD_CHUNK_SIZE` 就等于没有生效，语料会碎成大量没有上下文的残片。

所以 `ChunkService` 在交给 `TextChunker` 之前，先把**连续的同上下文小块**贪心打包到接近 `DOCGUARD_CHUNK_SIZE`：

| 条件 | 说明 |
| --- | --- |
| 块类型可索引 | `paragraph` / `list_item` / `table` / `table_row` / `caption`；`sentence` 和 `heading` 不入库 |
| `section_path` 相同 | 不跨标题合并，否则分块的章节引用会覆盖不到自己的全部内容 |
| `page_number` 相同 | 不跨页合并，页码引用同理 |
| 累计长度不超 `chunk_size` | 单块本身就超限时仍按原逻辑切开，不会被丢弃 |

打包后超过 `chunk_size` 的段落再由 `TextChunker` 按句号、问号、分号等自然边界切分，`DOCGUARD_CHUNK_OVERLAP` 在这一层生效。

用一份 35.9 万字的真实语料（33 份平台协议、功能说明与对外材料）实测：

| | 分块数 | 中位块长 | 平均块长 | 小于 50 字的块 |
| --- | --- | --- | --- | --- |
| 不合并 | 9639 | 21 字 | 37 字 | 7421（77.0%） |
| 合并后 | **1139** | **359 字** | **324 字** | **63（5.5%）** |

分块数直接决定 embedding 调用量、向量内存和 top-k 里的噪声，所以这一项对接真实语料前值得先确认。合并后没有任何分块超过 `chunk_size`。

两个需要知道的细节：

- **`block_ids` 只在内存和上传响应里完整。** `document_chunks` 表只有一个 `block_id` 列，存的是段落内第一个块的 id；重新读回时 `block_ids` 就只有这一个。要追溯完整成员关系得看上传时的返回值。
- **`start_offset` / `end_offset` 会映射回原文位置。** 合并会在块之间插入换行，拼接后的位置不再等于文档位置，因此分块时会按各成员块的原始偏移量还原。Markdown、TXT 这类解析器给了真实偏移量的格式，还原后是精确的；DOCX、PPTX 解析器不提供偏移量，此时保持 `TextChunker` 原有的段内相对值。这两个字段只用于引用定位，不参与打分。

已经入库的语料不会自动重切。改了 `DOCGUARD_CHUNK_SIZE` 或从旧版本升级后，用 `POST /v1/documents/{document_id}/reindex` 从原文件重新解析、分块、嵌入，见「重建索引」。

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

## 纯文字 AI 智能客服（通境三创 MVP）

初期客服采用纯文字交互：用户发送文字问题，系统返回文字答案和文档来源。不包含数字人、语音识别、语音合成、视频或头像渲染。`customer_service` 是一个基于知识库检索的轻量客服接口，提供：

- 简单意图识别：平台咨询、项目申报、账号问题、人工客服
- 基于文档的事实回答和来源引用
- 会话 `session_id`，对话历史落库、跨进程重启保留（见下文「会话历史」）
- 证据不足时拒答并转人工
- 本地 `DocGuard`、HTTP SDK 和 REST 三种调用方式

客服输入只接受 `message: str`；图片、DOCX、Markdown、PDF 等文件属于知识库建设阶段的数据源，不直接作为客服消息输入。

### Python 调用

```python
with DocGuard() as guard:
    answer = guard.customer_service.chat(
        "通境三创平台如何提交项目申报材料？",
        collection_id="tongjing-san-chuang",
        top_k=5,
    )
    print(answer.session_id)
    print(answer.intent, answer.status)
    print(answer.message)
    print(answer.handoff_required)

    # 继续同一会话
    follow_up = guard.customer_service.chat(
        "材料提交后在哪里查看进度？",
        session_id=answer.session_id,
        collection_id="tongjing-san-chuang",
    )

    # 清理会话上下文（软重置，返回该会话是否存在）
    found = guard.customer_service.reset(answer.session_id)
    print(found)
```

### HTTP SDK 调用

```python
with DocGuardClient("http://127.0.0.1:8000") as client:
    answer = client.customer_service.chat(
        "平台支持哪些创新创业服务？",
        collection_id="tongjing-san-chuang",
    )
    print(answer.message)
```

### REST 调用

```powershell
$body = @{
    message = "通境三创平台如何申报项目？"
    collection_id = "tongjing-san-chuang"
    top_k = 5
    handoff_if_unresolved = $true
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/v1/support/chat" `
  -ContentType "application/json" `
  -Body $body
```

客服返回的关键字段：

```text
session_id          会话标识，可用于继续追问
intent              platform / application / account / human / knowledge
status              ANSWERED / HANDOFF_REQUIRED / INSUFFICIENT_EVIDENCE
handoff_required    是否建议转人工
message             给用户展示的答复
sources             支撑回答的文档来源
suggested_actions   推荐的下一步操作
```

### 会话历史

会话历史落在 SQL 表 `support_sessions` 和 `support_messages` 里，不再保存在服务进程内存中。这意味着**重启进程、`uvicorn --workers N` 多进程、以及独立的 Celery worker 都能看到同一份历史**，也不会因为会话数增长而无限占用内存。

几点需要知道：

- **上限**：`DOCGUARD_SUPPORT_HISTORY_LIMIT`（默认 20 条消息，即 10 轮）。写入时就裁剪，最旧的静默丢弃，与早期 `deque(maxlen=20)` 的行为一致，所以表大小是 `会话数 × 20` 而不是随时间无限增长。
- **作用域是 `(tenant_id, session_id)` 而不是 `session_id` 单独**。调用方自带的 `session_id` 在两个租户间可能重复，复合唯一约束保证 A 租户既读不到也重置不了 B 租户的会话。
- **只有用户消息参与检索上下文**：`_contextual_query` 取最近 4 条 user 消息前置到当前问题，assistant 消息只作为记录保存。
- **已有数据库需要跑迁移**：`python -m docguard_rag.cli --migrate`，会应用 `0003_support_sessions`。全新库由 `init_database()` 直接建表，无需迁移。

`DELETE /v1/support/sessions/{session_id}` 是**软重置**：删除该会话的所有消息，保留会话行并标记 `status='RESET'`，留下"这个会话存在过并被重置"的审计痕迹。重置后同一个 `session_id` 可以继续使用，客户端会得到空历史——这正是预期语义。

对未知会话它**仍返回 200**，不给对接方引入新的失败模式；用响应体里的 `found` 区分：

```json
{"session_id": "cs_xxx", "status": "RESET", "found": true}
```

Python facade 的 `guard.customer_service.reset(...)` 和 HTTP SDK 的 `client.customer_service.reset(...)` 都返回这个信息（前者是 `bool`，后者是完整字典）。

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
| POST | `/v1/documents/reembed` | 用当前模型重嵌入语料 |
| GET | `/v1/documents/{document_id}/blocks` | 获取文档块 |
| GET | `/v1/documents/{document_id}/chunks` | 获取分块 |
| POST | `/v1/query` | RAG 查询 |
| POST | `/v1/correct` | 文本纠错 |
| POST | `/v1/reviews` | 创建审查 |
| GET | `/v1/reviews/{review_id}/issues` | 查询审查问题 |
| PATCH | `/v1/issues/{issue_id}` | 更新问题状态 |
| GET | `/v1/jobs/{job_id}` | 查询异步任务 |
| POST | `/v1/support/chat` | 智能客服对话（**数字人对接主入口**） |
| DELETE | `/v1/support/sessions/{session_id}` | 软重置会话历史 |
| GET | `/health` | 存活检查 |
| GET | `/ready` | 就绪检查 |

## 使用 Qdrant

默认向量后端是 `local`。本地模式不需要 Qdrant，SQL 表是向量与分块的唯一真相来源；检索读取一份进程内共享的只读语料快照（含 BM25 倒排索引），每次检索先用一条聚合查询比对指纹，只有语料发生变化时才重建快照，因此读多写少场景下检索开销不随查询次数累积。快照按数据库引擎共享，同进程内多个检索栈以及独立进程（如 Celery worker）写入后，下一次检索即可见。本地模式适合开发和小规模数据，更大规模请切换到 Qdrant。

### 向量检索的内存占用

本地模式把语料快照整个驻留在进程内存里，因此内存占用与「块数 × 向量维度」成正比。快照里有两份向量数据：一份是原始的 Python float 列表（`CorpusSnapshot.vectors`），另一份是检索时用的预归一化 `N × D` 矩阵（按维度分组）。实测每块占用：

| `DOCGUARD_EMBEDDING_DIMENSION` | Python float 列表 | 归一化矩阵 | 合计 |
| --- | --- | --- | --- |
| 128 | 4.49 KB/块 | 1.00 KB/块 | 5.49 KB/块 |
| 2048 | 64.49 KB/块 | 16.00 KB/块 | 80.49 KB/块 |

换算到常见规模：

| 块数 | 维度 128 | 维度 2048 |
| --- | --- | --- |
| 5 400 | ≈ 0.03 GiB | ≈ 0.41 GiB |
| 100 000 | ≈ 0.52 GiB | ≈ 7.68 GiB |

几点必须说清楚：

- **主要开销是 Python float 列表，不是矩阵。** 一个 Python float 占 24 字节、再加 8 字节列表指针，而矩阵里的 float64 只占 8 字节。矩阵化把检索从每查询数百毫秒降到毫秒级（2048 维、5400 块语料实测 194.9 ms → 1.65 ms），代价只是约 25% 的内存增量。
- **重嵌入期间会瞬时翻倍。** 矩阵按维度分组，`DOCGUARD_EMBEDDING_DIMENSION` 改过之后、语料重嵌入完成之前，新旧两种维度的向量同时存在于快照中，两种维度的查询各自命中自己那一批，不会出现检索空窗。跑完重嵌入并重启进程（或语料指纹变化触发重建）后回落。
- **进程数会成倍放大。** 用 `uvicorn --workers N` 或额外的 Celery worker 时，每个进程各持一份快照，上表要乘以进程数。

逃生口按推荐顺序：

1. **切到 Qdrant**（本节下文），向量不再驻留进程内存，这是既定的规模化路径。
2. **降低维度**。智谱 `embedding-3` 支持 256–2048 维，2048 与 1024 的检索质量差距通常小于 1024 与 128 的差距；把 `DOCGUARD_EMBEDDING_DIMENSION` 调低能线性削减内存，但改完必须重嵌入语料（见「重嵌入语料」）。
3. **拆分租户语料**，让单个进程的快照只覆盖它真正要检索的那部分。

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
# 必须等于 embedding 模型的实际输出宽度；改动后需要重建集合并重嵌入语料
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

### 智谱 GLM

智谱的 `/paas/v4` 端点与上述格式一致，可直接复制：

```dotenv
DOCGUARD_EMBEDDING_PROVIDER=compatible_api
DOCGUARD_EMBEDDING_BASE_URL=https://open.bigmodel.cn/api/paas/v4
DOCGUARD_EMBEDDING_API_KEY=<your-key>
DOCGUARD_EMBEDDING_MODEL_NAME=embedding-3
DOCGUARD_EMBEDDING_DIMENSION=2048

DOCGUARD_RERANKER_PROVIDER=compatible_api
DOCGUARD_RERANKER_BASE_URL=https://open.bigmodel.cn/api/paas/v4
DOCGUARD_RERANKER_API_KEY=<your-key>
DOCGUARD_RERANKER_MODEL_NAME=rerank

DOCGUARD_LLM_PROVIDER=compatible_api
DOCGUARD_LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
DOCGUARD_LLM_API_KEY=<your-key>
DOCGUARD_LLM_MODEL_NAME=glm-4-air
```

需要注意：

- **`DOCGUARD_EMBEDDING_DIMENSION` 必须等于模型的实际输出宽度**。`embedding-3` 支持 256-2048，默认 2048。配错时不会静默降级：每次入库和查询都会在写库前抛 `ProviderError`（HTTP 502），语料不受影响。
- **切换 embedding 模型或维度后必须重嵌入整个语料**，否则新旧向量宽度不一致，检索会拒绝旧向量。见「重嵌入语料」。
- `embedding-3` 单次请求最多 64 条文本、单条最多 3072 token。客户端按 `DOCGUARD_EMBEDDING_BATCH_SIZE`（默认 64）自动分批，调用方无需关心；任一批失败会中止整次入库并回滚，已写入的文件也会被删除。
- rerank 单次请求最多 128 篇、单篇最多 4096 字符。`HybridRetriever` 的融合候选在 `top_k=50` 时可达 400 篇，客户端会自动分批后按分数归并，结果与单次请求等价。
- 重嵌入或大批量入库会按批顺序发请求，**不做自动重试**。触发限流（429）时本次运行中止，已提交的批次保留，重新执行即可续做。

### 切到智谱后的验收清单

离线桩件下这些都无法验证，配好 `.env` 里的 key 之后按顺序跑一遍：

1. **向量真的建起来了。** 对既有文档调 `POST /v1/documents/{document_id}/reindex`，然后确认 `document_vectors` 行数等于 `document_chunks` 行数、且向量宽度是 2048：

   ```powershell
   sqlite3 docguard.db "SELECT (SELECT COUNT(*) FROM document_chunks), (SELECT COUNT(*) FROM document_vectors);"
   ```

   行数对不上说明有文档没重建；宽度不对说明 `DOCGUARD_EMBEDDING_DIMENSION` 和模型不匹配（这种情况会直接抛 502，不会静默）。

2. **语义召回真的生效了——这是本次切换的核心验收点。** 问一个**字面不同但语义相近**的问题，例如语料里写的是"如何注册账号"，你问"怎么开通登录凭证"。

   - 在 `hash` 离线 embedding 下这一条**必然失败**：`FakeEmbedding` 是字符 unigram/bigram 的哈希投影，没有任何语义理解，字面不重合就召回不到。
   - 切到 `embedding-3` 后应该能召回。如果还是召回不到，说明向量路没生效（检查 `document_vectors` 是否为空、维度是否匹配），而不是模型不行。

3. **查询缓存在工作。** 同一个问题连问两次，然后读计数器：

   ```python
   from docguard_rag.server.core.embedding_cache import query_embedding_cache
   print(query_embedding_cache().hits, query_embedding_cache().misses)
   ```

   注意第二次提问如果是走 `/v1/support/chat` 的**后续轮次**，查询串会带上对话历史前缀，属于合理 miss；要验证缓存请用 `/v1/query` 且 `corrective=false`。详见「查询向量缓存」。

4. **会话历史跨重启存活。** 用同一个 `session_id` 连发两轮 `POST /v1/support/chat`，**重启服务进程**，再发第三轮，确认它仍然看得到前两轮（可以问"我刚才问了什么"，或直接查 `support_messages` 表）。

5. **检索延迟没有劣化。** 2048 维下，几千块语料的单次检索应该在毫秒级（实测 5400 块 × 2048 维：1.65 ms）。如果明显更慢，先看快照是不是在每次查询都重建——正常情况下它被指纹门控，只在语料变化时重建一次：

   ```python
   from docguard_rag.server.core.retrieval_index import index_for_engine
   print(index_for_engine(engine).rebuilds)   # 连续查询两次，这个数字不应该变
   ```

   如果它每次都涨，说明指纹探测判定语料一直在变（通常是并发写入或时钟问题），那才是延迟来源；矩阵检索本身没有逐行回退路径。内存占用见「向量检索的内存占用」。

### 查询向量缓存

`DOCGUARD_EMBEDDING_QUERY_CACHE_SIZE`（默认 1024，设 0 关闭）控制一个进程级 LRU 缓存，命中时省掉一次 embedding 往返。键是 `(模型指纹, 查询文本)`：改过 `DOCGUARD_EMBEDDING_MODEL_NAME` 或 `DOCGUARD_EMBEDDING_DIMENSION` 之后所有键都变，旧条目自然失效，不需要手动清理。**与语料快照不同，这里没有失效信号，指纹本身就是失效机制**——所以缓存不会因为文档增删改而返回过期的向量。

缓存是进程级的，不按数据库引擎划分：查询向量只取决于模型和文本，与语料无关，而 `/v1/query` 与 `/v1/support/chat` 各自持有一套独立检索栈，共享缓存才能让两边的命中率叠加。

**别指望它能省掉「每个请求一次」的调用。** 实际收益取决于走哪条路径：

| 路径 | 是否命中 | 原因 |
| --- | --- | --- |
| Corrective RAG 第 1 次尝试 | 视重复度 | `QueryRewriter` 会先纠错，改写后的串与用户原文通常不同 |
| Corrective RAG 第 2 次尝试 | **必然 miss** | `rewrite(attempt=1)` 会剥掉「请问」「根据文档」等口语填充词，产出的是**另一个字符串**，这次 embedding 是必要的 |
| `/v1/support/chat` 首轮 | 视重复度 | 无历史时 `_contextual_query` 原样返回用户消息，跨会话的相同首问会命中 |
| `/v1/support/chat` 后续轮次 | **基本 miss** | 历史里最近 4 条用户消息会前置进查询串，每轮都不同 |
| `/v1/query` 且 `corrective=false` | 命中率高 | 直接用原文检索 |
| 健康检查、预热查询、压测重放 | 命中率高 | 同一串反复出现 |

也就是说 `corrective_max_retries=1` 时，**每个用户问题最多两次智谱往返**，缓存最多帮你省掉其中零次或一次；它真正解决的是重复问题、非 corrective 路径和无谓的预热开销，不是把每次查询都变成一次调用。

两个刻意的实现取舍：

- **计算在锁外进行。** 持锁跨一次 30 秒的 HTTP 调用会让进程内所有查询排在一个慢请求后面；代价是并发的相同 miss 会各自打一次网络——重复请求，不是错误答案。
- **失败不缓存。** provider 抛 `ProviderError`（超时、限流、维度不匹配）时异常在写入前就冒泡，下一次调用会真正重试，而不是重放上一次的失败。

命中率可以直接读：`from docguard_rag.server.core.embedding_cache import query_embedding_cache`，然后看 `query_embedding_cache().hits` / `.misses`。

## 重嵌入语料

向量是用「当时配置的模型」算出来的。换模型、换 provider、改 `DOCGUARD_EMBEDDING_DIMENSION` 之后，旧向量和新查询向量不在同一个语义空间里，检索结果会失去意义（宽度不同时更是直接报 502）。这时不需要重新解析文件——分块和块信息都还在库里，只把向量重算一遍即可。

三种调用方式，做的是同一件事：

```bash
# 1. CLI（运维最常用）。作用于 DOCGUARD_DATABASE_URL 指向的库，默认 ./docguard.db
python -m docguard_rag.cli --reembed
```

```python
# 2. Python facade
from docguard_rag import DocGuard

with DocGuard() as guard:
    result = guard.documents.reembed()                      # 整个租户
    result = guard.documents.reembed(document_ids=["doc_x"])  # 只重算指定文档
    result = guard.documents.reembed(async_mode=True)        # 大语料走任务队列
    print(result.reembedded, result.batches, result.job_id)
```

```bash
# 3. REST
curl -X POST http://127.0.0.1:8000/v1/documents/reembed \
  -H 'Content-Type: application/json' \
  -d '{"document_ids": [], "batch_size": null, "async": false}'
```

`async: true` 会立即返回 `job_id`，进度可用 `GET /v1/jobs/{job_id}` 查询；同步模式则在请求里跑完，大语料会超时，务必用异步。

**失败与续做**：每批（默认 `DOCGUARD_EMBEDDING_BATCH_SIZE` 条）单独提交一次事务。中途报错或被限流时，已提交的批次是有效的，返回值和 job 里的计数只统计真正完成的部分；直接重跑即可，重复执行是幂等的。刻意不做自动重试，是为了让 429 立刻暴露而不是把限流拖成几十分钟的静默等待。

**返回值里要留意的一项**：`embedded_without_offsets` 表示「这个分块在库里原本就没有向量行」。这种块照样会被嵌入、照样能检索，但 `start_offset` / `end_offset` 无法恢复——它们由分块在块内的位置推导，从来没落库。这两个字段只用于引用定位展示，不参与打分。如果需要精确保真，对这些文档改用 `POST /v1/documents/{document_id}/reindex`（返回值的 `document_ids_needing_reindex` 会列出它们），它会从原文件重新解析，偏移量完整。

> 当前库如果是从「向量表为空」的状态起步（例如只入库过、但入库时 provider 报错回滚），走 `reindex` 比 `reembed` 更合适：它能一次性把偏移量也补齐。

**内存代价**：重嵌入过程中，旧维度和新维度的向量会短暂共存，常驻内存约翻倍。见「向量检索的内存占用」。

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

启用认证后，文档、检索、审查、问题、任务和客服会话接口都会执行租户过滤。客服会话的作用域是 `(tenant_id, session_id)`，所以即使两个租户使用同一个 `session_id`，也读不到、重置不了对方的历史。

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

测试套件与本地配置隔离，`tests/conftest.py` 在收集用例之前完成：

- `DOCGUARD_DATABASE_URL` 指向一次性临时库，不会读写仓库里的 `docguard.db`
- `DOCGUARD_UPLOAD_DIR` 指向同一临时目录
- embedding / reranker / LLM / 向量库 / 任务后端全部钉在离线桩件上，即使 `.env` 里配了真实模型密钥也不会发起网络请求

以上都用 `setdefault` 写入，所以在 shell 里显式导出同名环境变量即可覆盖（例如跑真实模型的集成验证）。`tests/test_system.py::test_pytest_binds_an_isolated_database` 是哨兵：一旦被指向仓库默认库就会失败。

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

### 报 `Embedding dimension mismatch`（HTTP 502）

`DOCGUARD_EMBEDDING_DIMENSION` 与模型实际输出宽度不一致。这是**故意设计成硬失败**的：宽度写错的向量入库后永远检索不到，静默降级比报错更难查。

改法：把 `DOCGUARD_EMBEDDING_DIMENSION` 设成模型的真实输出宽度（智谱 `embedding-3` 默认 2048，可选 256-2048），然后重嵌入语料：

```powershell
python -m docguard_rag.cli --reembed
```

因为校验发生在写库之前，报错时语料还没有被污染，改完配置直接重跑即可。

### 检索报 `does not match any indexed vector dimension`（HTTP 502）

改过 `DOCGUARD_EMBEDDING_DIMENSION` 但语料还是旧维度的向量。同上，跑一次 `--reembed` 或对单个文档调 `POST /v1/documents/{document_id}/reindex`。

重嵌入是分批提交的，过程中新旧两种维度会共存，此时两种维度的查询各自命中自己那一批，**不会出现检索空窗**。

### 语义检索像是没生效（换个说法就搜不到）

按可能性排序检查：

1. **`document_vectors` 是空的。** 只有分块没有向量，混合检索就退化成纯 BM25 关键词匹配，字面不重合必然召回不到。跑 `--reembed` 或逐文档 `reindex`。
2. **还在用离线 `hash` provider。** 确认 `.env` 里 `DOCGUARD_EMBEDDING_PROVIDER=compatible_api` 而不是默认的 `hash`。`FakeEmbedding` 是字符哈希投影，**没有任何语义理解**，在它下面"怎么注册账号"和"如何开通登录凭证"就是两个不相干的串。
3. **维度配错**，见上一条。

验证方法见「切到智谱后的验收清单」第 2 项。

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

自定义 provider 还可以实现两个**可选**协议，都不是抽象方法，所以既有实现无需改动：

| 属性/方法 | 适用于 | 作用 | 不实现的后果 |
| --- | --- | --- | --- |
| `fingerprint` | embedding / reranker / LLM | 标识"这是哪个模型"，作为查询向量缓存的键的一部分 | 该 provider 的查询**不进缓存**，每次都打网络 |
| `close()` | 三者 | 释放 HTTP 连接，由 FastAPI lifespan 统一调用 | 连接在进程退出时才释放 |

`fingerprint` 返回一个可哈希的元组即可，惯例是 `(provider_kind, model_name, dimension)`。改过模型名或维度后所有缓存键都变，旧条目自然失效——这是查询缓存唯一需要的失效机制。
