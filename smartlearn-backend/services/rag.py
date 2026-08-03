"""RAG 管线工具 —— 文本清洗、PDF 抽取、分块、JSON 存取、预览、嵌入"""

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


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

def extract_pages_for_rag(file_path: str | Path) -> list[dict]:
    """从路径读取 PDF，返回清洗后的 {page, text} 列表，保留原始页码"""
    file_path = Path(file_path)
    reader = PdfReader(str(file_path))

    records: list[dict] = []
    for page_number, page in enumerate(reader.pages, start=1):
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


# ── 6c. 调度器 ───────────────────────────────────────────

def build_chunks(
    records: list[dict],
    chunk_mode: str,
    chunk_size: int,
    overlap: int = 0,
) -> list[dict]:
    """根据 chunk_mode 选择分块策略，返回统一 schema 的 chunk 列表

    chunk_mode:
        - "paragraph"        段落边界分块，保留完整性
        - "character"        固定窗口，无重叠
        - "character_overlap" 固定窗口，邻块共享 overlap 个字符
    """
    if chunk_mode == "paragraph":
        return chunk_by_paragraph(records, chunk_size)

    if chunk_mode == "character":
        return chunk_by_characters(records, chunk_size, overlap=0)

    if chunk_mode == "character_overlap":
        if overlap <= 0:
            raise ValueError("character_overlap 模式需要 overlap > 0")
        return chunk_by_characters(records, chunk_size, overlap=overlap)

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
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = 32,
) -> np.ndarray:
    """将文本列表编码为归一化的 float32 向量矩阵"""
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

