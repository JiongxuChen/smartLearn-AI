import os

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from services.llm import answer_from_pages
from services.pdf import (
    PDFError,
    PDFInvalidError,
    PDFNoTextError,
    PDFTooLargeError,
    extract_pages,
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

# 临时内存存储：chat_id → 解析后的 PDF 页面
chat_store: dict[str, list[dict]] = {}


class ChatRequest(BaseModel):
    message: str


@app.get("/")
async def root():
    return {"message": "SmartLearn Lite API is running"}


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/upload")
async def upload_pdf(chat_id: str, file: UploadFile = File(...)):
    """上传 PDF 文件，解析并存入临时内存，通过 chat_id 关联"""
    pdf_bytes = await file.read()

    try:
        pages = extract_pages(pdf_bytes)
    except PDFTooLargeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except PDFInvalidError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except PDFNoTextError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except PDFError as e:
        raise HTTPException(status_code=400, detail=str(e))

    chat_store[chat_id] = pages
    return {"chat_id": chat_id, "pages": pages}


@app.post("/chat")
async def chat(chat_id: str, body: ChatRequest):
    """基于已上传 PDF 的智能问答，答案引用具体页码"""
    pages = chat_store.get(chat_id)
    if pages is None:
        raise HTTPException(status_code=404, detail="chat_id 未找到，请先上传 PDF")

    answer = answer_from_pages(pages, body.message)
    return {"chat_id": chat_id, "answer": answer}
