export default function PdfPreview({ chatId, targetPage, jumpKey }) {
  if (!chatId) {
    return <div className="pdf-placeholder">请先上传 PDF 文件</div>;
  }

  const base = import.meta.env.VITE_API_URL || "http://localhost:8000";
  const src = `${base}/documents/${chatId}/file${targetPage ? `#page=${targetPage}` : ""}`;

  return (
    <div className="pdf-preview">
      <iframe
        key={jumpKey}
        src={src}
        title="PDF 预览"
        className="pdf-iframe"
      />
    </div>
  );
}
