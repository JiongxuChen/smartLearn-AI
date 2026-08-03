# SmartLearn RAG 管线

基于 FAISS + sentence-transformers 的本地 RAG 管线，支持 PDF 文本提取、多种分块策略、向量检索、多轮 LLM 问答和 Chroma 附录分支。

## 架构总览

```
┌── Python Notebook ──────────────────────┐
│  services/rag.py  ← 所有核心逻辑        │
│  ├─ §1-5  文本清洗 & PDF 抽取 & JSON    │
│  ├─ §6    分块策略（4 种模式）           │
│  ├─ §7    嵌入管线（sentence-transformers）│
│  ├─ §8    FAISS 索引（默认检索路径）     │
│  ├─ §9    检索 & 本地答案                │
│  ├─ §10   Wrapper / Server 接口          │
│  ├─ §11   检索评估                       │
│  ├─ §12   Chroma 附录分支                │
│  └─ §13   Web 服务层接口                 │
└──────────────────────────────────────────┘
         │
         │  import
         ▼
┌── FastAPI Server (main.py) ─────────────┐
│  GET  /                                 │
│  GET  /health                           │
│  POST /upload?chat_id=   → RAG 档案构建 │
│  POST /chat              → 多轮检索+回答│
│  GET  /documents/{id}/file → PDF 回传   │
└──────────────────────────────────────────┘
         │
         │  HTTP fetch
         ▼
┌── React Frontend ───────────────────────┐
│  App.jsx         上传状态 & 工作区分栏  │
│  PdfPreview.jsx  iframe 预览 + 跳页     │
│  ChatPanel.jsx   多轮消息 & citation    │
│  api.js          HTTP 封装              │
└──────────────────────────────────────────┘
```

## 数据流

```
PDF 文件
  ↓  extract_pages_for_rag() / extract_pages_from_bytes_for_rag()
Page 记录 [{page, text}, ...]
  ↓  build_chunks() / chunk_with_langchain_recursive()
Chunk 记录 [{chunk_id, page, text, ...}, ...]
  ↓  embed_texts()
Embeddings (N, D) float32
  ↓  build_faiss_index()
FAISS Index (IndexIDMap + IndexFlatIP)
  ↓  search_document()
Top-K Hits [{page, chunk_id, text, score}, ...]
  ↓  answer_document() — 有 API key → LLM；无 → best_sentence_answer
{answer, citations, sources}
  ↓  answer_chat_turn() — 自动追加 history，多轮检索带上下文
```

## 目录结构

```
smartLearn-AI/
├── smartlearn-backend/
│   ├── main.py                    # FastAPI 路由
│   ├── Dockerfile                 # Railway 部署
│   ├── services/
│   │   ├── rag.py                 # RAG 管线核心（~1400 行）
│   │   ├── pdf.py                 # PDF 解析 & 异常类
│   │   └── llm.py                 # 原始 LLM 调用（已被 rag.py 替代）
│   ├── artifacts/rag/             # RAG 产物缓存
│   │   ├── {document_id}/
│   │   │   ├── pages_{doc}.json
│   │   │   ├── chunks_{mode}_{size}.json
│   │   │   ├── embeddings_{mode}_{size}_{model}.npy
│   │   │   ├── index_{mode}_{size}_{model}.faiss
│   │   │   ├── index_{mode}_{size}_{model}.json
│   │   │   └── manifest.json
│   │   ├── chroma/                # Chroma 持久化（附录）
│   │   ├── hf_models/             # 本地模型缓存
│   │   └── reports/
│   └── uploads/                   # 上传的原始 PDF
├── smartlearn-frontend/
│   └── src/
│       ├── App.jsx                # 上传 + 分栏工作区
│       ├── PdfPreview.jsx         # PDF iframe 预览
│       ├── ChatPanel.jsx          # 多轮对话面板
│       ├── api.js                 # 后端 API 封装
│       └── index.css              # 全局样式
└── test_files/                    # 测试用 PDF
```

---

## API 路由

| 方法 | 路径 | 说明 | 响应 |
|------|------|------|------|
| `GET` | `/` | 服务状态 | `{"message":"SmartLearn Lite API is running"}` |
| `GET` | `/health` | 健康检查 | `{"ok":true}` |
| `POST` | `/upload?chat_id=<id>` | 上传 PDF，构建 RAG 档案 | `{"status":"ok","filename":"...","pages":11,"characters":32584}` |
| `POST` | `/chat?chat_id=<id>` | RAG 问答（检索 + LLM + 记录历史） | `{"chat_id":"...","answer":"...","citations":[2,3],"sources":[...]}` |
| `GET` | `/documents/{chat_id}/file` | 返回原始 PDF | `application/pdf` |

---

## 模块功能

### §1 — 文本清洗
| 函数 | 说明 |
|------|------|
| `clean_text(text)` | 去空字节、软连字符、断字连字符（`-\n`）、行内换行→空格，压缩多余空白 |

### §2 — PDF 抽取
| 函数 | 说明 |
|------|------|
| `extract_pages_for_rag(file_path, page_limit=None)` | 从文件路径读取 PDF，逐页清洗，保留原始页码 |
| `extract_pages_from_bytes_for_rag(pdf_bytes)` | 从 bytes 读取（供上传路由用），同管线 |

### §3–4 — JSON 存取
| 函数 | 说明 |
|------|------|
| `save_json(data, file_path)` | UTF-8 保存，自动创建父目录 |
| `load_json(file_path)` | UTF-8 读取 |

### §5 — 快速预览
| 函数 | 说明 |
|------|------|
| `preview_records(records, columns, n=5)` | 打印前 N 条指定列的表格 |

### §6 — 分块策略
| 函数 | chunk_mode | 特点 |
|------|-----------|------|
| `chunk_by_paragraph(records, chunk_size)` | `paragraph` | `\n\n` 断段落，超长在空格断开 |
| `chunk_by_characters(records, chunk_size, overlap=0)` | `character` / `character_overlap` | 固定窗口，overlap>0 滑动重叠 |
| `chunk_with_langchain_recursive(pages, chunk_size, chunk_overlap)` | `langchain_recursive` | 递归：`\n\n` → `\n` → ` ` → 字符 |
| `build_chunks(records, chunk_mode, chunk_size, overlap=0)` | — | 调度器 |

### §7 — 嵌入管线

| 函数 | 说明 |
|------|------|
| `get_device()` | 返回 `"cpu"` 或 `"cuda"` |
| `load_model(model_name)` | 加载 SentenceTransformer，进程级缓存 |
| `embed_texts(model, texts, batch_size, ...)` | 批量编码 → 归一化 float32。支持两种调用方式 |
| `ensure_artifacts(document_id, ...)` | pages → chunks → embeddings，签名匹配走缓存 |

```python
# embed_texts 双签名
v = embed_texts(model, ["句子1"])                              # 旧风格：传模型对象
v = embed_texts(["句子1"], model_name="...all-MiniLM-L6-v2")   # 新风格：传模型名
```

### §8 — FAISS 索引

| 函数 | 说明 |
|------|------|
| `build_faiss_index(embeddings)` | 归一化向量 → `IndexIDMap(IndexFlatIP)`（内积=余弦） |
| `save_faiss_index` / `load_faiss_index` | 读写 `.faiss` 文件 |
| `ensure_index(document_id, ...)` | pages→chunks→embeddings→index，签名匹配走缓存 |
| `prepare_rag_document(document_id, filename, pages, ...)` | 返回完整文档记录 |

### §9 — 检索 & 本地答案

| 函数 | 说明 |
|------|------|
| `search_bundle(question, bundle, top_k, candidate_pool)` | 内存 index + chunks 检索 |
| `search_document(question, document, top_k, candidate_pool)` | 从 prepared document 加载索引检索 |
| `split_sentences(text)` | 按 `. ! ?` 拆句 |
| `best_sentence_answer(question, hits)` | 词重叠最多的句子 + `(Page N)` |

### §10 — Wrapper / Server 接口

| 函数 | 说明 |
|------|------|
| `extract_citations(answer, hits)` | 解析 `[Page N]` |
| `build_sources(hits)` | `{page, chunk_id, score, preview}` |
| `answer_document(document, question, ...)` | **完整 RAG**：检索 → LLM（或本地回退）。**有历史时把上一轮问答注入检索查询** |
| `append_history(document, question, result)` | 追加到 `document["history"]` |

```python
result = answer_document(doc, "Transformer 有多少层?")
# → {"answer": "6 层 [Page 2]", "citations": [2], "sources": [...]}
```

**多轮对话**：第二问 "Give one more detail from that page" 时，检索查询自动变成：
```
第一问 — 第一轮答案…
Give one more detail from that page.
```
embedding 带着上下文去搜，指代消解无需 LLM 参与。

### §11 — 检索评估

| 函数 | 说明 |
|------|------|
| `evaluate_questions(eval_set, documents_by_name, ...)` | 批量评估，返回 DataFrame（`retrieval_hit` + `answer_hit`） |
| `normalize_for_match(text)` | 标准化用于子串匹配 |
| `contains_any_answer(text, answers)` | 子串命中判断 |

### §12 — Chroma 附录分支

FAISS 的平行替代。集合存储在 `artifacts/rag/chroma/{document_id}/`。

| 函数 | 说明 |
|------|------|
| `build_chroma_collection(...)` | 构建 Chroma 集合 |
| `query_chroma_collection(...)` | 相似度查询 |
| `search_document_with_chroma(...)` | Chroma 检索 |
| `answer_document_with_chroma(...)` | Chroma 问答 |
| `ensure_artifact_dirs(...)` | 创建全部产物目录 |

### §13 — Web 服务层

供 `main.py` 调用的入口函数：

| 函数 | 调用方 | 说明 |
|------|--------|------|
| `prepare_rag_chat_record(chat_id, filename, pdf_bytes, pages, ...)` | POST /upload | 保存 PDF → 抽取 → 构建 RAG 档案 → 返回 `documents[chat_id]` |
| `build_upload_response(document)` | POST /upload | `{chat_id, filename, page_count, chunk_count}` |
| `build_grounded_user_prompt(question, hits, history)` | answer_document | 拼装 grounded prompt（chunks + 历史） |
| `answer_document_turn(document, question, ...)` | notebook | answer_document + 自动追加 history |
| `answer_chat_turn(document, message, ...)` | POST /chat | 路由入口：检索 → LLM → 记录历史 |

---

## 前端组件

### App.jsx
- 管理共享状态：`uploaded`、`currentPage`、`pdfJumpKey`、`chatKey`
- 上传框始终可见，上传后显示"已加载 N 页 PDF"
- 上传成功后 `chatKey + 1` → ChatPanel 完全重挂载（旧对话清零）
- `handleJumpToPage(page)` → 更新 `currentPage` + 触发 iframe 重载

### PdfPreview.jsx
- `<iframe key={jumpKey} src="/documents/{chatId}/file#page=N">` 
- `jumpKey` 每次点击 citation 递增 → iframe 重挂载 → PDF 跳到对应页

### ChatPanel.jsx
- 自管 `messages`、`loading`、`error`
- `enabled` prop 变化时 `useEffect` 清空消息
- 内部调用 `askQuestion(message, chatId)`
- citation chip 按钮 → `onJumpToPage(page)`

### api.js
```js
uploadPDF(file, chatId)    → POST /upload
askQuestion(message, chatId) → POST /chat
getPdfUrl(chatId)           → /documents/{chatId}/file
```

---

## 默认配置

| 参数 | 值 |
|------|-----|
| `chunk_mode` | `character_overlap` |
| `chunk_size` | 700 |
| `overlap` | 120 |
| `model_name` | `sentence-transformers/all-MiniLM-L6-v2` |
| `artifact_root` | `artifacts/rag` |
| `top_k` | 3 |
| `candidate_pool` | 60 |
| `answer_model` | `openrouter/free` |

## 缓存机制

`ensure_artifacts` / `ensure_index` 基于 MD5 签名（document_id + page count + chunk_mode + chunk_size + overlap + model_name）：

- PDF 变了 → 全部重算
- chunk 参数变了 → chunks + embeddings + index 重算
- 模型变了 → embeddings + index 重算
- 什么都没变 → 读磁盘缓存，秒出

## 快速开始

```bash
# 1. 启动后端
cd smartlearn-backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# 2. 启动前端
cd smartlearn-frontend
npm install
npm run dev

# 3. 打开 http://localhost:5173
```

## 依赖

```
faiss-cpu        sentence-transformers
pypdf            numpy            pandas
torch            langchain-text-splitters  # 可选
chromadb                                      # 可选
fastapi          uvicorn          python-multipart
openai           python-dotenv
```
