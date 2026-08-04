"""RAG 管线工具 —— 文本清洗、PDF 抽取、分块、JSON 存取、预览、嵌入"""

import faiss
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    _LC_AVAILABLE = True
except ImportError:
    RecursiveCharacterTextSplitter = None  # type: ignore[assignment]
    _LC_AVAILABLE = False


# ── 1. 文本清洗 ──────────────────────────────────────────

def clean_text(text: str) -> str:
    """清洗单页 PDF 提取文本：去空字节、软连字符、断字换行、多余空白"""
    # 去掉空字节和软连字符
    text = text.replace("\x00", "")
    text = text.replace("­", "")

    # 合并断字换行（"consump-\ntion" → "consumption"）
    text = re.sub(r"-\n\s*", "", text)

    # 统一换行符
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 单换行 → 空格（PDF 内部换行即同一段落）
    # 双换行保留为段落分隔
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # 折叠多余空白
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ── 2. PDF 逐页抽取 ──────────────────────────────────────

def extract_pages_for_rag(
    file_path: str | Path, page_limit: int | None = None
) -> list[dict]:
    """从路径读取 PDF，返回清洗后的 {page, text} 列表，保留原始页码

    Args:
        file_path: PDF 文件路径
        page_limit: 可选，限制读取页数（方便快速测试）
    """
    file_path = Path(file_path)
    reader = PdfReader(str(file_path))

    records: list[dict] = []
    for page_number, page in enumerate(reader.pages, start=1):
        if page_limit is not None and page_number > page_limit:
            break
        raw = page.extract_text() or ""
        cleaned = clean_text(raw)
        if cleaned:  # 跳过空页
            records.append({"page": page_number, "text": cleaned})

    return records


# ── 3. JSON 存 ───────────────────────────────────────────

def save_json(data, file_path: str | Path) -> None:
    """将 Python 对象保存为 UTF-8 JSON，必要时创建父目录"""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ── 4. JSON 取 ───────────────────────────────────────────

def load_json(file_path: str | Path):
    """从 UTF-8 JSON 文件读取回 Python 对象"""
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


# ── 5. 快速预览 ──────────────────────────────────────────

def preview_records(records: list[dict], columns: list[str], n: int = 5) -> None:
    """打印前 n 条记录的指定列，列宽自适应"""
    head = records[:n]
    if not head:
        print("(空)")

    # 计算每列最大宽度
    widths = {}
    for col in columns:
        cell_widths = [len(str(row.get(col, ""))) for row in head]
        cell_widths.append(len(col))
        widths[col] = max(cell_widths)

    # 表头
    header = " │ ".join(col.ljust(widths[col]) for col in columns)
    print(header)
    print("─" * len(header))

    # 数据行
    for row in head:
        line = " │ ".join(
            str(row.get(col, "")).ljust(widths[col]) for col in columns
        )
        print(line)

    print(f"\n（共 {len(records)} 条，显示前 {len(head)} 条）")


# ── 6. 分块工具 ──────────────────────────────────────────

def slice_long_text(text: str, chunk_size: int) -> list[str]:
    """将超长文本在最近的空格处切开，尽量避免断词"""
    pieces: list[str] = []
    remaining = text
    while len(remaining) > chunk_size:
        # 在 chunk_size 范围内找最后一个空格
        cut = remaining[:chunk_size].rfind(" ")
        if cut == -1:
            cut = chunk_size  # 无空格，硬切
        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def _make_chunk(page: int, idx: int, text: str, chunk_mode: str) -> dict:
    """构造统一 schema 的 chunk 字典"""
    return {
        "chunk_id": f"p{page}_c{idx}",
        "page": page,
        "text": text,
        "chunk_mode": chunk_mode,
    }


# ── 6a. 段落分块 ─────────────────────────────────────────

def chunk_by_paragraph(records: list[dict], chunk_size: int) -> list[dict]:
    """按 \n\n 段落边界分块，超长段落用 slice_long_text 子切"""
    chunks: list[dict] = []

    for rec in records:
        page = rec["page"]
        paragraphs = rec["text"].split("\n\n")
        idx = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(para) <= chunk_size:
                chunks.append(_make_chunk(page, idx, para, "paragraph"))
                idx += 1
            else:
                # 段落超长 → 在空格处切开
                for piece in slice_long_text(para, chunk_size):
                    chunks.append(_make_chunk(page, idx, piece, "paragraph"))
                    idx += 1

    return chunks


# ── 6b. 固定窗口分块 ─────────────────────────────────────

def chunk_by_characters(
    records: list[dict], chunk_size: int, overlap: int = 0
) -> list[dict]:
    """固定大小滑动窗口分块，overlap > 0 时邻块共享 overlap 个字符"""
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) 必须小于 chunk_size ({chunk_size})")

    chunk_mode = "character_overlap" if overlap > 0 else "character"
    step = chunk_size - overlap
    chunks: list[dict] = []

    for rec in records:
        page = rec["page"]
        text = rec["text"]
        start = 0
        idx = 0

        while start < len(text):
            piece = text[start : start + chunk_size]
            if not piece.strip():
                start += step
                continue

            chunk = _make_chunk(page, idx, piece, chunk_mode)
            # 有重叠时标注关联邻居
            if overlap > 0 and idx > 0:
                chunk["overlap_with"] = f"p{page}_c{idx - 1}"
            chunks.append(chunk)

            start += step
            idx += 1

    return chunks


# ── 6c. LangChain 递归分块 ────────────────────────────────

def chunk_with_langchain_recursive(
    pages: list[dict],
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str] | None = None,
) -> list[dict]:
    """用 RecursiveCharacterTextSplitter 逐页切分，处理噪声段落边界

    separators 优先级默认：双换行 → 单换行 → 空格 → 字符回退
    """
    if not _LC_AVAILABLE:
        raise ImportError(
            "langchain_text_splitters 未安装。请执行：\n"
            "  pip install langchain-text-splitters"
        )
    if separators is None:
        separators = ["\n\n", "\n", " ", ""]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        keep_separator=True,
    )

    chunks: list[dict] = []
    for rec in pages:
        page = rec["page"]
        pieces = splitter.split_text(rec["text"])
        for idx, piece in enumerate(pieces):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                {
                    "chunk_id": f"p{page}_c{idx}",
                    "page": page,
                    "text": piece,
                    "chunk_mode": "langchain_recursive",
                }
            )

    return chunks


# ── 6d. 调度器 ───────────────────────────────────────────

def build_chunks(
    records: list[dict],
    chunk_mode: str,
    chunk_size: int,
    overlap: int = 0,
) -> list[dict]:
    """根据 chunk_mode 选择分块策略，返回统一 schema 的 chunk 列表

    chunk_mode:
        - "paragraph"           段落边界分块，保留完整性
        - "character"           固定窗口，无重叠
        - "character_overlap"   固定窗口，邻块共享 overlap 个字符
        - "langchain_recursive" 递归分割器，优先段落→句子→空格→字符
    """
    if chunk_mode == "paragraph":
        return chunk_by_paragraph(records, chunk_size)

    if chunk_mode == "character":
        return chunk_by_characters(records, chunk_size, overlap=0)

    if chunk_mode == "character_overlap":
        if overlap <= 0:
            raise ValueError("character_overlap 模式需要 overlap > 0")
        return chunk_by_characters(records, chunk_size, overlap=overlap)

    if chunk_mode == "langchain_recursive":
        return chunk_with_langchain_recursive(records, chunk_size, overlap)

    raise ValueError(f"未知的 chunk_mode: {chunk_mode!r}")


# ═══════════════════════════════════════════════════════════
# ── 7. 嵌入管线 ──────────────────────────────────────────
# ═══════════════════════════════════════════════════════════

# ── 7a. 模型名 → 安全文件名后缀 ──────────────────────────

def model_tag(model_name: str) -> str:
    """将模型名转为安全文件名后缀："/" → "_" """
    return model_name.replace("/", "_")


# ── 7b. 优先本地缓存 ────────────────────────────────────

def resolve_model_source(model_name: str) -> str:
    """优先返回本地缓存模型路径，不存在则返回原始名称"""
    # 1. 直接给的路径已存在
    if os.path.isdir(model_name):
        return model_name

    # 2. 模型简称（路径最后一段，如 "all-MiniLM-L6-v2"）
    short = model_name.rsplit("/", 1)[-1] if "/" in model_name else model_name

    # 3. 检查 artifacts/rag/hf_models/ 下多种命名
    base = Path("artifacts") / "rag" / "hf_models"
    for candidate in [model_tag(model_name), short]:
        local_dir = base / candidate
        if local_dir.is_dir():
            return str(local_dir)

    # 4. sentence-transformers 内置缓存
    hub_cache = Path.home() / ".cache" / "torch" / "sentence_transformers"
    for candidate in [model_tag(model_name), short]:
        cached = hub_cache / candidate
        if cached.is_dir():
            return str(cached)

    # 5. 回退：让库从 HuggingFace 下载
    return model_name


# ── 7c. 设备选择 ────────────────────────────────────────

def get_device() -> str:
    """返回当前机器可用的最佳设备：cuda 或 cpu"""
    return "cuda" if torch.cuda.is_available() else "cpu"


# ── 7d. 加载模型（缓存复用） ─────────────────────────────

_model_cache: dict[str, SentenceTransformer] = {}


def load_model(model_name: str) -> SentenceTransformer:
    """创建或复用 SentenceTransformer 实例"""
    if model_name in _model_cache:
        return _model_cache[model_name]

    source = resolve_model_source(model_name)
    device = get_device()
    model = SentenceTransformer(source, device=device)
    _model_cache[model_name] = model
    return model


# ── 7e. 文本批量编码 ────────────────────────────────────

def embed_texts(
    texts_or_model: list[str] | SentenceTransformer | str,
    texts_or_batch_size: list[str] | int | None = None,
    batch_size: int = 32,
    model_cache_dir: str | Path | None = None,
    model_name: str | None = None,
) -> np.ndarray:
    """将文本列表编码为归一化的 float32 向量矩阵

    两种调用方式均支持：
    1. 旧风格 embed_texts(model, texts, batch_size=32)
    2. 新风格 embed_texts(texts, model_name="...", model_cache_dir="...", batch_size=1)
    """
    # ── 检测调用模式 ─────────────────────────────────
    if isinstance(texts_or_model, list):
        # 新风格：第一个参数是 texts
        texts = texts_or_model
        # model_name 来自 kwarg 或第二个位置参数
        resolved_model_name = model_name or texts_or_batch_size
        if resolved_model_name is None:
            raise ValueError("需要 model_name 参数")
        if isinstance(resolved_model_name, int):
            raise TypeError("第二个参数应为模型名，收到了 int — 请用 model_name= kwarg")
        model = load_model(str(resolved_model_name))
    else:
        # 旧风格：第一个参数是 model 对象或模型名
        model = texts_or_model
        texts = texts_or_batch_size
        if isinstance(model, str):
            model = load_model(model)

    if batch_size is None:
        batch_size = 32
    if isinstance(texts_or_batch_size, int) and texts is None:
        raise TypeError("请用旧风格 embed_texts(model, texts) 或新风格 embed_texts(texts, model_name=...)")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings.astype(np.float32)


# ── 7f. 产物路径决议 ────────────────────────────────────

def artifact_paths_for(
    document_id: str,
    chunk_mode: str,
    model_name: str,
    chunk_size: int,
    overlap: int = 0,
    artifact_root: str | Path = "artifacts/rag",
) -> dict:
    """返回 pages/chunks/embeddings/manifest 的保存路径"""
    root = Path(artifact_root) / document_id
    tag = model_tag(model_name)

    # chunk 文件名含模式+大小
    if overlap > 0:
        chunk_stem = f"chunks_{chunk_mode}_{chunk_size}_o{overlap}"
        emb_stem = f"embeddings_{chunk_mode}_{chunk_size}_o{overlap}_{tag}"
    else:
        chunk_stem = f"chunks_{chunk_mode}_{chunk_size}"
        emb_stem = f"embeddings_{chunk_mode}_{chunk_size}_{tag}"

    return {
        "root": root,
        "pages": root / f"pages_{document_id}.json",
        "chunks": root / f"{chunk_stem}.json",
        "embeddings": root / f"{emb_stem}.npy",
        "index": root / f"index_{chunk_stem}_{tag}.faiss",
        "index_meta": root / f"index_{chunk_stem}_{tag}.json",
        "manifest": root / "manifest.json",
    }


# ── 7g. 编排器：pages → chunks → embeddings → manifest ───

def ensure_artifacts(
    document_id: str,
    pdf_name: str,
    pages: list[dict],
    chunk_mode: str,
    model_name: str,
    chunk_size: int,
    overlap: int = 0,
    batch_size: int = 32,
    artifact_root: str | Path = "artifacts/rag",
) -> dict:
    """构建或复用 pages/chunks/embeddings 产物，签名匹配时走缓存

    Returns:
        {"pages": [...], "chunks": [...], "embeddings": np.ndarray, "manifest": {...}}
    """
    paths = artifact_paths_for(
        document_id, chunk_mode, model_name, chunk_size, overlap, artifact_root,
    )

    # ── 签名校验 ──────────────────────────────────────
    sig = hashlib.md5(
        json.dumps(
            {
                "document_id": document_id,
                "pdf_name": pdf_name,
                "num_pages": len(pages),
                "chunk_mode": chunk_mode,
                "chunk_size": chunk_size,
                "overlap": overlap,
                "model_name": model_name,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()

    # manifest 存在且签名匹配 → 完全走缓存
    if paths["manifest"].exists():
        try:
            cached = load_json(paths["manifest"])
            if cached.get("sig") == sig:
                # 读回 pages
                if paths["pages"].exists():
                    pages = load_json(paths["pages"])
                # 读回 chunks
                chunks = load_json(paths["chunks"]) if paths["chunks"].exists() else []
                # 读回 embeddings
                if paths["embeddings"].exists():
                    embeddings = np.load(paths["embeddings"])
                else:
                    embeddings = np.empty((0, 0), dtype=np.float32)

                return {
                    "pages": pages,
                    "chunks": chunks,
                    "embeddings": embeddings,
                    "manifest": cached,
                }
        except Exception:
            pass  # 缓存损坏 → 重新计算

    # ── 逐步检查缓存 / 计算 ───────────────────────────
    # Pages
    if not paths["pages"].exists():
        save_json(pages, paths["pages"])

    # Chunks
    if paths["chunks"].exists():
        chunks = load_json(paths["chunks"])
    else:
        chunks = build_chunks(pages, chunk_mode, chunk_size, overlap)
        save_json(chunks, paths["chunks"])

    # Embeddings
    if paths["embeddings"].exists():
        embeddings = np.load(paths["embeddings"])
    else:
        model = load_model(model_name)
        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(model, texts, batch_size)
        np.save(paths["embeddings"], embeddings)

    # ── 写入 manifest ─────────────────────────────────
    device = get_device()
    manifest = {
        "document_id": document_id,
        "pdf_name": pdf_name,
        "num_pages": len(pages),
        "chunk_mode": chunk_mode,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "model_name": model_name,
        "num_chunks": len(chunks),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.size else 0,
        "device": device,
        "chunk_path": str(paths["chunks"]),
        "embedding_path": str(paths["embeddings"]),
        "raw_pages_path": str(paths["pages"]),
        "sig": sig,
    }
    save_json(manifest, paths["manifest"])

    return {
        "pages": pages,
        "chunks": chunks,
        "embeddings": embeddings,
        "manifest": manifest,
    }


# ═══════════════════════════════════════════════════════════
# ── 8. FAISS 索引 ────────────────────────────────────────
# ═══════════════════════════════════════════════════════════

# ── 8a. 路径显示工具 ──────────────────────────────────────

def relative_path_str(path: str | Path, base: str | Path) -> str:
    """返回相对于 base 的短路径，方便 notebook 输出"""
    try:
        return str(Path(path).relative_to(base))
    except ValueError:
        return str(path)


# ── 8b. 构建 FAISS 索引 ──────────────────────────────────

def build_faiss_index(embeddings: np.ndarray) -> "faiss.Index":
    """用归一化嵌入构建 FAISS 内积索引（等价余弦相似度搜索）

    Args:
        embeddings: 形状 (N, D) 的归一化 float32 向量
    Returns:
        faiss.IndexIDMap(IndexFlatIP) 可搜索索引
    """
    dim = embeddings.shape[1]
    # IndexFlatIP：内积搜索，归一化向量下等价于余弦相似度
    index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
    ids = np.arange(embeddings.shape[0], dtype=np.int64)
    index.add_with_ids(embeddings, ids)
    return index


# ── 8c. 保存索引 ─────────────────────────────────────────

def save_faiss_index(index: "faiss.Index", index_path: str | Path) -> None:
    """将 FAISS 索引写入二进制 .faiss 文件"""
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))


# ── 8d. 加载索引 ─────────────────────────────────────────

def load_faiss_index(index_path: str | Path) -> "faiss.Index":
    """从 .faiss 文件读回 FAISS 索引"""
    return faiss.read_index(str(index_path))


# ── 8e. 编排器：pages → chunks → embeddings → index ──────

def ensure_index(
    document_id: str,
    pdf_name: str,
    pages: list[dict] | None = None,
    pdf_path: str | Path | None = None,
    chunk_mode: str = "character_overlap",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 700,
    overlap: int = 120,
    batch_size: int = 32,
    artifact_root: str | Path | None = None,
) -> dict:
    """构建或复用 FAISS 索引，签名匹配时走缓存

    至少需要 pages 或 pdf_path 之一。
    """
    if artifact_root is None:
        artifact_root = "artifacts/rag"

    # 如果没给 pages，从 PDF 抽取
    if pages is None:
        if pdf_path is None:
            raise ValueError("需要 pages 或 pdf_path 至少提供一个")
        pages = extract_pages_for_rag(pdf_path)

    # 复用 ensure_artifacts 拿 chunks + embeddings
    bundle = ensure_artifacts(
        document_id, pdf_name, pages,
        chunk_mode, model_name, chunk_size, overlap,
        batch_size, artifact_root,
    )
    chunks = bundle["chunks"]
    embeddings = bundle["embeddings"]
    manifest = bundle["manifest"]

    paths = artifact_paths_for(
        document_id, chunk_mode, model_name, chunk_size, overlap, artifact_root,
    )

    # 签名匹配 + 索引文件存在 → 加载
    if paths["index"].exists() and paths["index_meta"].exists():
        cached_meta = load_json(paths["index_meta"])
        if cached_meta.get("sig") == manifest["sig"]:
            index = load_faiss_index(paths["index"])
            return {
                "pages": pages,
                "chunks": chunks,
                "embeddings": embeddings,
                "manifest": manifest,
                "index": index,
                "index_path": str(paths["index"]),
                "index_meta_path": str(paths["index_meta"]),
            }

    # 构建索引
    index = build_faiss_index(embeddings)
    save_faiss_index(index, paths["index"])

    # 保存索引元数据（含签名以匹配缓存）
    index_meta = {
        "document_id": document_id,
        "chunk_mode": chunk_mode,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "model_name": model_name,
        "embedding_dim": int(embeddings.shape[1]),
        "num_vectors": embeddings.shape[0],
        "sig": manifest["sig"],
    }
    save_json(index_meta, paths["index_meta"])

    return {
        "pages": pages,
        "chunks": chunks,
        "embeddings": embeddings,
        "manifest": manifest,
        "index": index,
        "index_path": str(paths["index"]),
        "index_meta_path": str(paths["index_meta"]),
    }


# ── 8f. Server 风格文档记录 ──────────────────────────────

def prepare_rag_document(
    document_id: str,
    filename: str,
    pages: list[dict],
    chunk_mode: str = "character_overlap",
    chunk_size: int = 700,
    overlap: int = 120,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    artifact_root: str | Path | None = None,
) -> dict:
    """将 PDF 页面转为带 FAISS 索引的 server 风格文档记录

    Returns:
        {
            "document_id": str,
            "filename": str,
            "pages": [...],
            "chunks": [...],
            "chunk_size": int,
            "embedding_dim": int,
            "history": [],          # 预留：未来对话历史
            "index": {              # RAG 索引元数据
                "model": str,
                "chunk_mode": str,
                "chunk_size": int,
                "overlap": int,
                "num_chunks": int,
                "embedding_dim": int,
                "faiss_path": str,
            },
            "artifacts": {          # 产物路径（方便 notebook 展示）
                "pages": str,
                "chunks": str,
                "index": str,
                "manifest": str,
            },
        }
    """
    if artifact_root is None:
        artifact_root = "artifacts/rag"

    bundle = ensure_index(
        document_id=document_id,
        pdf_name=filename,
        pages=pages,
        chunk_mode=chunk_mode,
        model_name=model_name,
        chunk_size=chunk_size,
        overlap=overlap,
        batch_size=batch_size,
        artifact_root=artifact_root,
    )

    paths = artifact_paths_for(
        document_id, chunk_mode, model_name, chunk_size, overlap, artifact_root,
    )

    return {
        "document_id": document_id,
        "filename": filename,
        "pages": pages,
        "chunks": bundle["chunks"],
        "chunk_size": chunk_size,
        "embedding_dim": bundle["manifest"]["embedding_dim"],
        "model_name": model_name,
        "model_source": resolve_model_source(model_name),
        "history": [],
        "index": {
            "model": model_name,
            "chunk_mode": chunk_mode,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "num_chunks": len(bundle["chunks"]),
            "embedding_dim": bundle["manifest"]["embedding_dim"],
            "faiss_path": bundle["index_path"],
        },
        "artifacts": {
            "pages": str(paths["pages"]),
            "chunks": str(paths["chunks"]),
            "embeddings": str(paths["embeddings"]),
            "index": bundle["index_path"],
            "manifest": str(paths["manifest"]),
        },
    }


# ═══════════════════════════════════════════════════════════
# ── 9. 检索 & 本地答案 ───────────────────────────────────
# ═══════════════════════════════════════════════════════════

# ── 9a. 关键词集合 ───────────────────────────────────────

def keyword_set(text: str) -> set[str]:
    """从文本提取轻量词汇集合（≥3 字符，去标点，小写）"""
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return {w for w in cleaned.split() if len(w) >= 3}


# ── 9b. 对内存 bundle 检索 ───────────────────────────────

def search_bundle(
    question: str,
    bundle: dict,
    top_k: int = 3,
    candidate_pool: int = 60,
    batch_size: int = 1,
    history: list[dict] | None = None,
) -> list[dict]:
    """用内存中的 index + chunks 检索 top-k 命中

    Args:
        question: 用户问题
        bundle: ensure_index 返回的 bundle（含 index, chunks, manifest）
        top_k: 最终返回的命中数
        candidate_pool: FAISS 初检候选数（越大越精确但越慢）
        batch_size: 嵌入时批大小
        history: 未使用，预留参数
    """
    index: "faiss.Index" = bundle["index"]
    chunks: list[dict] = bundle["chunks"]
    manifest: dict = bundle["manifest"]

    # 1. 嵌入问题（复用同一模型）
    model = load_model(manifest["model_name"])
    q_vec = embed_texts(model, [question], batch_size=batch_size)
    # 确保 (1, D) 形状
    if q_vec.ndim == 1:
        q_vec = q_vec.reshape(1, -1)

    # 2. FAISS 检索
    n_candidates = min(candidate_pool, index.ntotal)
    distances, ids = index.search(q_vec, n_candidates)

    # 3. 问题关键词
    q_keywords = keyword_set(question)

    # 4. 组装命中
    hits: list[dict] = []
    for i in range(len(ids[0])):
        chunk_idx = int(ids[0][i])
        if chunk_idx < 0 or chunk_idx >= len(chunks):
            continue
        score = float(distances[0][i])
        chunk = chunks[chunk_idx]
        kw_overlap = len(q_keywords & keyword_set(chunk["text"]))

        hits.append(
            {
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "score": round(score, 4),
                "keyword_overlap": kw_overlap,
            }
        )

    # 5. 按 score 降序取 top_k
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits[:top_k]


# ── 9c. 对文档记录检索（从磁盘加载索引） ─────────────────

def search_document(
    question: str,
    document: dict,
    top_k: int = 3,
    candidate_pool: int = 60,
    history: list[dict] | None = None,
) -> list[dict]:
    """从 prepared document 记录加载 FAISS 索引并检索

    Args:
        question: 用户问题
        document: prepare_rag_document 返回的记录
        top_k: 返回命中数
        candidate_pool: FAISS 初检候选数
        history: 未使用，预留参数
    """
    index_path = document["artifacts"]["index"]
    chunks = document["chunks"]
    index_meta = document["index"]

    # 加载索引
    index = load_faiss_index(index_path)

    # 构建临时 bundle
    bundle = {
        "index": index,
        "chunks": chunks,
        "manifest": {
            "model_name": index_meta["model"],
            "embedding_dim": index_meta["embedding_dim"],
        },
    }

    return search_bundle(
        question, bundle,
        top_k=top_k, candidate_pool=candidate_pool,
        history=history,
    )


# ── 9d. 拆句 ─────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    """将文本按句号、问号、感叹号拆分为候选句子"""
    # 在 . ! ? 后跟空格/换行/结尾处切分
    raw = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in raw if s.strip()]


# ── 9e. 最佳句子答案 ─────────────────────────────────────

def best_sentence_answer(question: str, hits: list[dict]) -> str:
    """从检索命中里选与问题关键词重叠最多的句子，带页码标签

    Returns:
        如 "The Transformer uses 6 encoder layers …… (Page 3)"
    """
    if not hits:
        return "（未找到相关内容）"

    q_keywords = keyword_set(question)
    best_sentence = ""
    best_page = hits[0]["page"]
    best_score = -1

    for hit in hits:
        for sentence in split_sentences(hit["text"]):
            overlap = len(q_keywords & keyword_set(sentence))
            if overlap > best_score:
                best_score = overlap
                best_sentence = sentence
                best_page = hit["page"]

    if not best_sentence:
        # 无关键词命中 → 返回 FAISS 得分最高的 chunk 摘要 + 提示
        top = hits[0]
        preview = top["text"][:150]
        return (
            f"（本地搜索未找到精确匹配，以下为语义最相似的片段，建议配置 OPENROUTER_API_KEY 以获得准确回答）\n\n"
            f"{preview} (Page {top['page']}, score={top['score']:.2f})"
        )

    return f"{best_sentence} (Page {best_page})"


# ═══════════════════════════════════════════════════════════
# ── 10. Wrapper / Server 接口 ────────────────────────────
# ═══════════════════════════════════════════════════════════

# ── 10a. 引用提取 ────────────────────────────────────────

def extract_citations(
    answer: str, hits: list[dict] | None = None
) -> list[int]:
    """从回答中提取 [Page N] / [Pages N-M] 引用，去重排序

    若回答中没有匹配到页码，回退到 hits 中出现过的页码。
    """
    re_pages = r"\[Pages?\s*(\d+)(?:\s*[-–]\s*(\d+))?\]"
    pages: set[int] = set()

    for m in re.finditer(re_pages, answer):
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else start
        for p in range(start, end + 1):
            pages.add(p)

    if pages:
        return sorted(pages)

    # 回退：用检索命中的页码
    if hits:
        return sorted({h["page"] for h in hits})

    return []


# ── 10b. 前端来源列表 ────────────────────────────────────

def build_sources(hits: list[dict]) -> list[dict]:
    """将检索命中转为前端友好的来源对象"""
    return [
        {
            "page": h["page"],
            "chunk_id": h["chunk_id"],
            "score": h["score"],
            "preview": h["text"][:120],
        }
        for h in hits
    ]


# ── 10c. 文档问答 ────────────────────────────────────────

def answer_document(
    document: dict,
    question: str,
    top_k: int = 3,
    candidate_pool: int = 60,
    answer_model: str = "openrouter/free",
) -> dict:
    """对一份已准备好的文档提问，检索 + 回答

    - 有 OpenRouter API key → 用 LLM 对检索 chunks + 对话历史回答
    - 无 API key → 本地 best_sentence_answer 作为轻量回退
    """
    history = document.get("history", [])

    # 有历史时，把上一轮问题拼入检索查询（只用问题，避免长答案淹没新问题）
    search_query = question
    if history:
        prev = history[-1]
        prev_q = prev.get("question", "")
        if prev_q:
            search_query = f"{prev_q}\n{question}"

    # 检索
    hits = search_document(
        search_query, document,
        top_k=top_k, candidate_pool=candidate_pool,
    )

    sources = build_sources(hits)

    # 尝试 LLM 回答
    answer = None
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        try:
            from openai import OpenAI

            prompt = build_grounded_user_prompt(question, hits, history)
            client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
            )
            response = client.chat.completions.create(
                model=answer_model,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = response.choices[0].message.content or ""
        except Exception as e:
            import traceback
            err_name = type(e).__name__
            print(f"[LLM ERROR] {err_name}: {e}")
            traceback.print_exc()
            # 限流时在 answer 里给用户提示
            if "429" in str(e) or "Rate limit" in str(e):
                answer = (
                    "（OpenRouter 免费额度已用完，明天自动重置。"
                    "可在配置面板中更换 LLM 模型或充值。）"
                )

    # 回退：本地抽取
    if answer is None:
        answer = best_sentence_answer(question, hits)

    citations = extract_citations(answer, hits)

    return {
        "answer": answer,
        "citations": citations,
        "sources": sources,
    }


# ── 10d. 追加对话历史 ────────────────────────────────────

def append_history(
    document: dict, question: str, result: dict
) -> list[dict]:
    """将一轮问答追加到 document["history"] 并返回更新后的列表"""
    entry = {
        "question": question,
        "answer": result["answer"],
        "citations": result["citations"],
        "sources": result["sources"],
    }
    document["history"].append(entry)
    return document["history"]


# ═══════════════════════════════════════════════════════════
# ── 11. 检索评估 ────────────────────────────────────────
# ═══════════════════════════════════════════════════════════

# ── 11a. 文本标准化（用于子串匹配）───────────────────────

def normalize_for_match(text: str) -> str:
    """小写 + 去标点 + 压缩空白 → 用于字符串级别比对"""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)   # 标点 → 空格
    text = re.sub(r"\s+", " ", text)       # 折叠空白
    return text.strip()


# ── 11b. 子串命中检查 ───────────────────────────────────

def contains_any_answer(text: str, answers: list[str]) -> bool:
    """text 标准化后是否包含任一 gold answer 的子串"""
    norm_text = normalize_for_match(text)
    for a in answers:
        if normalize_for_match(a) in norm_text:
            return True
    return False


# ── 11c. 批量评估 ───────────────────────────────────────

def evaluate_questions(
    eval_set: list[dict],
    documents_by_name: dict[str, dict],
    top_k: int = 3,
    candidate_pool: int = 60,
    answers_key: str = "acceptable_answers",
) -> "pd.DataFrame":
    """逐条评估检索 + 本地答案的质量

    eval_set 每条需含:
        - question: str
        - 由 answers_key 指定的 gold answer 列表（默认 "acceptable_answers"）
        - pdf_name: str         ← 对应 documents_by_name 的 key

    Returns:
        DataFrame，每行含 question, pages, local_answer,
        retrieved_chunks_preview, retrieval_hit, answer_hit
    """
    import pandas as pd

    rows: list[dict] = []

    for item in eval_set:
        question = item["question"]
        gold_answers = item[answers_key]
        pdf_name = item["pdf_name"]

        doc = documents_by_name.get(pdf_name)
        if doc is None:
            rows.append(
                {
                    "question": question,
                    "pages": [],
                    "local_answer": f"[文档 {pdf_name!r} 未找到]",
                    "retrieved_chunks_preview": "",
                    "retrieval_hit": False,
                    "answer_hit": False,
                }
            )
            continue

        # 检索
        hits = search_document(
            question, doc, top_k=top_k, candidate_pool=candidate_pool,
        )
        retrieved_pages = sorted({h["page"] for h in hits})
        answer = best_sentence_answer(question, hits)

        # 判断两个 hit
        retrieval_hit = any(
            contains_any_answer(h["text"], gold_answers) for h in hits
        )
        answer_hit = contains_any_answer(answer, gold_answers)

        # 预览：取得分最高的 chunk 前 150 字
        preview = hits[0]["text"][:150] if hits else ""

        rows.append(
            {
                "question": question,
                "pages": retrieved_pages,
                "local_answer": answer,
                "retrieved_chunks_preview": preview,
                "retrieval_hit": retrieval_hit,
                "answer_hit": answer_hit,
            }
        )

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════
# ── 12. Chroma 附录分支（FAISS 的平行替代） ─────────────
# ═══════════════════════════════════════════════════════════

# ── 12a. 懒加载 chromadb ─────────────────────────────────

def _require_chromadb():
    """返回 chromadb 模块，未安装时给出明确提示"""
    try:
        import chromadb
        return chromadb
    except ImportError:
        raise ImportError(
            "chromadb 未安装。请执行：\n"
            "  pip install chromadb"
        )


# ── 12b. 产物目录 ────────────────────────────────────────

def ensure_artifact_dirs(
    artifact_root: str | Path | None = None,
) -> dict[str, Path]:
    """创建并返回全部产物目录（含 Chroma 子目录）"""
    root = Path(artifact_root or "artifacts/rag")
    dirs = {
        "root": root,
        "raw_pages": root / "raw_pages",
        "chunks": root / "chunks",
        "embeddings": root / "embeddings",
        "indexes": root / "indexes",
        "reports": root / "reports",
        "chroma": root / "chroma",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# ── 12c. 构建 Chroma 集合 ────────────────────────────────

def build_chroma_collection(
    document_id: str,
    chunks: list[dict],
    embeddings: np.ndarray,
    persist_dir: str | Path,
) -> dict:
    """用 chunks + embeddings 构建 Chroma 向量集合

    Args:
        document_id: 唯一文档 ID（作为 collection 名称）
        chunks: chunk 记录列表
        embeddings: (N, D) float32 归一化向量
        persist_dir: Chroma 持久化根目录

    Returns:
        {"collection_name": str, "item_count": int, "persist_dir": str}
    """
    chromadb = _require_chromadb()

    # Chroma 集合名要求 3-512 字符，只含 [a-zA-Z0-9._-]
    safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", document_id)
    if len(safe_name) < 3:
        safe_name = f"doc_{safe_name}"

    client = chromadb.PersistentClient(path=str(persist_dir))

    # 同名 collection 已存在则删除重建
    try:
        client.delete_collection(safe_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=safe_name,
        metadata={"hnsw:space": "cosine"},
    )

    # 更新 document_id 为安全名（后续查询用同一名称）
    document_id = safe_name

    # 准备数据
    ids = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {"page": c["page"], "chunk_id": c["chunk_id"]}
        for c in chunks
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=metadatas,
    )

    return {
        "collection_name": document_id,
        "item_count": len(chunks),
        "persist_dir": str(persist_dir),
    }


# ── 12d. 查询 Chroma 集合 ────────────────────────────────

def query_chroma_collection(
    document_id: str,
    query_embedding: np.ndarray,
    persist_dir: str | Path,
    top_k: int,
) -> list[dict]:
    """对 Chroma 集合执行相似度查询

    Returns:
        [{chunk_id, page, text, score}, ...] 按 score 降序
    """
    chromadb = _require_chromadb()

    # 与 build 一致的名称清洗
    safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", document_id)
    if len(safe_name) < 3:
        safe_name = f"doc_{safe_name}"

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(safe_name)

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits: list[dict] = []
    ids_list = results.get("ids", [[]])[0]
    docs_list = results.get("documents", [[]])[0]
    metas_list = results.get("metadatas", [[]])[0]
    dists_list = results.get("distances", [[]])[0]

    for i in range(len(ids_list)):
        # Chroma cosine distance → 转为 cosine similarity (1 - distance)
        dist = float(dists_list[i]) if i < len(dists_list) else 0.0
        score = round(1.0 - dist, 4)
        meta = metas_list[i] if i < len(metas_list) else {}
        text = docs_list[i] if i < len(docs_list) else ""

        hits.append(
            {
                "chunk_id": meta.get("chunk_id", ids_list[i]),
                "page": int(meta.get("page", 0)),
                "text": text,
                "score": score,
            }
        )

    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits


# ── 12e. Chroma 检索 ─────────────────────────────────────

def search_document_with_chroma(
    question: str,
    document: dict,
    persist_dir: str | Path,
    top_k: int = 3,
    batch_size: int = 1,
) -> list[dict]:
    """用 Chroma 检索 top-k 命中，返回与 FAISS 检索一致的 shape

    Returns:
        [{page, chunk_id, text, score}, ...]
    """
    model = load_model(document["index"]["model"])
    q_vec = embed_texts(model, [question], batch_size=batch_size)
    if q_vec.ndim == 1:
        q_vec = q_vec.reshape(1, -1)

    return query_chroma_collection(
        document["document_id"],
        q_vec,
        persist_dir,
        top_k,
    )


# ── 12f. Chroma 回答 ─────────────────────────────────────

def answer_document_with_chroma(
    document: dict,
    question: str,
    persist_dir: str | Path,
    top_k: int = 3,
    answer_model: str = "openrouter/free",
) -> dict:
    """Chroma 路径的文档问答，返回 {answer, citations, sources}

    与 answer_document 完全一致的返回 shape。
    """
    hits = search_document_with_chroma(
        question, document, persist_dir, top_k=top_k,
    )

    sources = build_sources(hits)

    # 尝试 LLM
    answer = None
    try:
        from services.llm import answer_from_pages

        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key:
            answer = answer_from_pages(hits, question)
    except Exception:
        pass

    if answer is None:
        answer = best_sentence_answer(question, hits)

    citations = extract_citations(answer, hits)

    return {
        "answer": answer,
        "citations": citations,
        "sources": sources,
    }


# ═══════════════════════════════════════════════════════════
# ── 13. Web 服务层接口 ───────────────────────────────────
# ═══════════════════════════════════════════════════════════

# ── 13a. bytes → pages（供上传路由使用）───────────────────

def extract_pages_from_bytes_for_rag(pdf_bytes: bytes) -> list[dict]:
    """从 PDF 字节流提取清洗后的 {page, text} 列表

    与 extract_pages_for_rag 使用同一套 clean_text 管线。
    """
    from io import BytesIO
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(pdf_bytes))
    records: list[dict] = []
    for page_number, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        cleaned = clean_text(raw)
        if cleaned:
            records.append({"page": page_number, "text": cleaned})
    return records


# ── 13b. 上传时构建服务器文档记录 ─────────────────────────

def prepare_rag_chat_record(
    chat_id: str,
    filename: str,
    pdf_bytes: bytes | None = None,
    pages: list[dict] | None = None,
    upload_root: str | Path | None = None,
    chunk_mode: str = "character_overlap",
    chunk_size: int = 700,
    overlap: int = 120,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    artifact_root: str | Path | None = None,
) -> dict:
    """上传时一步构建 documents[chat_id] 记录

    保存 PDF 到磁盘 → 抽取页面 → 构建 RAG 档案。
    pdf_bytes 和 pages 至少提供一个。
    """
    if pages is None and pdf_bytes is None:
        raise ValueError("需要 pdf_bytes 或 pages 至少提供一个")

    if upload_root is None:
        upload_root = "uploads"

    # 保存 PDF
    upload_dir = Path(upload_root) / chat_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / filename
    if pdf_bytes is not None:
        file_path.write_bytes(pdf_bytes)

    # 抽取页面
    if pages is None:
        pages = extract_pages_from_bytes_for_rag(pdf_bytes)

    # 委托 prepare_rag_document 构建 RAG 档案
    doc = prepare_rag_document(
        document_id=chat_id,
        filename=filename,
        pages=pages,
        chunk_mode=chunk_mode,
        chunk_size=chunk_size,
        overlap=overlap,
        model_name=model_name,
        batch_size=batch_size,
        artifact_root=artifact_root,
    )

    # 补充上传专有字段
    doc["saved_pdf_path"] = str(file_path)
    return doc


# ── 13c. 前端上传响应 ─────────────────────────────────────

def build_upload_response(document: dict) -> dict:
    """从服务端文档记录提取上传成功响应"""
    return {
        "chat_id": document["document_id"],
        "filename": document["filename"],
        "page_count": len(document["pages"]),
        "chunk_count": len(document["chunks"]),
    }


# ── 13d. 构建 grounded prompt ─────────────────────────────

def build_grounded_user_prompt(
    question: str,
    hits: list[dict],
    history: list[dict] | None = None,
) -> str:
    """用检索到的 chunks + 可选对话历史构建 grounded prompt"""
    chunks_text = "\n\n".join(
        f"[Page {h['page']}] {h['text']}" for h in hits
    )

    history_text = ""
    if history:
        recent = history[-3:]  # 只取最近 3 轮避免过长
        turns = "\n".join(
            f"用户: {t['question']}\n助手: {t['answer']}" for t in recent
        )
        history_text = f"最近对话：\n{turns}\n\n"

    return (
        "你只能根据下方 PDF 片段回答问题。如果片段中找不到答案，"
        "请明确说「文档未提供足够信息」。引用时使用 [Page N] 标记。\n\n"
        f"{history_text}"
        f"PDF 片段：\n{chunks_text}\n\n"
        f"问题：{question}"
    )


# ── 13e. 带对话历史的问答 ────────────────────────────────

def answer_document_turn(
    document: dict,
    question: str,
    top_k: int = 3,
    candidate_pool: int = 60,
    answer_model: str = "openrouter/free",
) -> dict:
    """answer_document + 自动追加 history，返回结果含更新后的历史"""
    result = answer_document(
        document, question,
        top_k=top_k, candidate_pool=candidate_pool,
        answer_model=answer_model,
    )
    result["history"] = append_history(document, question, result)
    return result


# ── 13f. POST /chat 路由级接口 ────────────────────────────

def answer_chat_turn(
    document: dict,
    message: str,
    top_k: int = 3,
    candidate_pool: int = 60,
    answer_model: str = "openrouter/free",
) -> dict:
    """路由层问答入口：检索 → LLM → 记录历史 → 返回

    供 main.py POST /chat 路由调用，
    返回 {answer, citations, sources}。
    """
    return answer_document_turn(
        document, message,
        top_k=top_k, candidate_pool=candidate_pool,
        answer_model=answer_model,
    )
