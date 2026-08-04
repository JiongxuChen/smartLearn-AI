import os

from dotenv import load_dotenv
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from services.rag import (
    answer_chat_turn,
    extract_pages_from_bytes_for_rag,
    prepare_rag_chat_record,
)

load_dotenv()

app = FastAPI()

# ── CORS 配置 ────────────────────────────────────────────
allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 临时内存存储：chat_id → RAG 文档记录（含 pages, chunks, 索引, 对话历史）
documents: dict[str, dict] = {}


class ChatRequest(BaseModel):
    message: str
    top_k: int | None = None
    candidate_pool: int | None = None
    answer_model: str | None = None
    retrieval_backend: str | None = None  # "faiss" / "chroma"


@app.get("/")
async def root():
    return {"message": "SmartLearn Lite API is running"}


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/upload")
async def upload_pdf(
    chat_id: str,
    file: UploadFile = File(...),
    chunk_mode: str | None = None,
    chunk_size: int | None = None,
    overlap: int | None = None,
    model_name: str | None = None,
):
    """上传 PDF 文件，构建 RAG 档案并存入内存"""
    pdf_bytes = await file.read()
    filename = file.filename or "upload.pdf"

    # 校验是否为有效 PDF（通过 pypdf 解析）
    try:
        pages = extract_pages_from_bytes_for_rag(pdf_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="无法识别的文件格式，请上传 PDF 文件")

    if not pages:
        raise HTTPException(
            status_code=422,
            detail="无法从该 PDF 中提取文字（可能是扫描件或空文件）",
        )

    # 所有页面文字为空 → 扫描件
    total_chars = sum(len(p["text"]) for p in pages)
    if total_chars == 0:
        raise HTTPException(
            status_code=422,
            detail="该 PDF 为扫描件或图片，无文字层。OCR 功能暂不支持",
        )

    try:
        kwargs = {}
        if chunk_mode is not None: kwargs["chunk_mode"] = chunk_mode
        if chunk_size is not None: kwargs["chunk_size"] = chunk_size
        if overlap is not None: kwargs["overlap"] = overlap
        if model_name is not None: kwargs["model_name"] = model_name

        doc = prepare_rag_chat_record(
            chat_id=chat_id,
            filename=filename,
            pdf_bytes=pdf_bytes,
            pages=pages,
            **kwargs,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"RAG 档案构建失败：{e}")

    documents[chat_id] = doc

    return {
        "status": "ok",
        "filename": filename,
        "pages": len(pages),
        "characters": total_chars,
    }


@app.post("/chat")
async def chat(chat_id: str, body: ChatRequest):
    """基于已上传 PDF 的智能问答，检索 + LLM / 本地回退"""
    doc = documents.get(chat_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="chat_id 未找到，请先上传 PDF")

    top_k = body.top_k or 3
    candidate_pool = body.candidate_pool or 60
    answer_model = body.answer_model or "openrouter/free"

    # 按检索后端选择路径
    if body.retrieval_backend == "chroma":
        from services.rag import (
            answer_document_with_chroma,
            build_chroma_collection,
            ensure_artifact_dirs,
        )
        import numpy as np
        chroma_dir = ensure_artifact_dirs()["chroma"]
        # 每次切换都重建 Chroma 集合（build 自带覆盖逻辑）
        emb_path = doc.get("artifacts", {}).get("embeddings", "")
        if emb_path:
            embeddings = np.load(emb_path)
        else:
            from services.rag import embed_texts
            texts = [c["text"] for c in doc["chunks"]]
            embeddings = embed_texts(
                doc.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
                texts,
                batch_size=32,
            )
        build_chroma_collection(
            doc["document_id"], doc["chunks"],
            embeddings, chroma_dir,
        )
        result = answer_document_with_chroma(
            doc, body.message, chroma_dir,
            top_k=top_k, answer_model=answer_model,
        )
    else:
        result = answer_chat_turn(
            doc, body.message,
            top_k=top_k,
            candidate_pool=candidate_pool,
            answer_model=answer_model,
        )

    return {
        "chat_id": chat_id,
        "answer": result["answer"],
        "citations": result.get("citations", []),
        "sources": result.get("sources", []),
    }


@app.post("/chat/reset")
async def reset_chat(chat_id: str):
    """清空对话历史，保留已上传的 PDF 档案"""
    doc = documents.get(chat_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="chat_id 未找到")
    doc["history"] = []
    return {"status": "ok"}


@app.get("/documents/{chat_id}/file")
async def get_pdf_file(chat_id: str):
    """返回已上传 PDF 的原始文件"""
    doc = documents.get(chat_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="chat_id 未找到")

    file_path = Path(doc.get("saved_pdf_path", ""))
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="PDF 文件不存在")

    return FileResponse(str(file_path), media_type="application/pdf")
