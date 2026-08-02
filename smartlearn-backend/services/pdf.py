"""PDF 解析服务 —— 自定义异常 + extract_pages"""

from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError

MAX_PAGES = 30


# ── 自定义异常 ────────────────────────────────────────────

class PDFError(Exception):
    """PDF 处理异常的基类"""


class PDFInvalidError(PDFError):
    """无法识别的文件格式或文件损坏"""


class PDFTooLargeError(PDFError):
    """PDF 页数超过限制"""


class PDFNoTextError(PDFError):
    """扫描件 / 无文字层，不支持 OCR"""


# ── 核心函数 ──────────────────────────────────────────────

def extract_pages(pdf_bytes: bytes) -> list[dict]:
    """解析 PDF 字节流，返回每页文本。

    Raises:
        PDFInvalidError: 空文件 / 非 PDF / 损坏
        PDFTooLargeError: 超过 MAX_PAGES 页
        PDFNoTextError: 全部页面无文字（扫描件）
    """
    # ── 1. 空文件检测 ─────────────────────────────────
    if not pdf_bytes:
        raise PDFInvalidError("文件为空，请上传有效的 PDF 文件")

    # ── 2. 解析 PDF ────────────────────────────────────
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
    except PdfReadError:
        raise PDFInvalidError("无法识别的文件格式，请上传 PDF 文件")
    except Exception:
        raise PDFInvalidError("无法识别的文件格式，请上传 PDF 文件")

    # ── 3. 页数限制 ────────────────────────────────────
    if len(reader.pages) > MAX_PAGES:
        raise PDFTooLargeError(f"PDF 最多允许 {MAX_PAGES} 页，当前为 {len(reader.pages)} 页")

    # ── 4. 提取文本 ────────────────────────────────────
    pages = [
        {
            "page": page_number,
            "text": (page.extract_text() or "").strip(),
        }
        for page_number, page in enumerate(reader.pages, start=1)
    ]

    # ── 5. 扫描件检测（全部页面无文字）────────────────
    if all(p["text"] == "" for p in pages):
        raise PDFNoTextError(
            "该 PDF 为扫描件或图片，无文字层。OCR 功能暂不支持"
        )

    return pages
