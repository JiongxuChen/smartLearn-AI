import { useState } from "react";
import { uploadPDF } from "./api";
import PdfPreview from "./PdfPreview";
import ChatPanel from "./ChatPanel";
import ConfigPanel from "./ConfigPanel";

const CHAT_ID = "day3-demo";

const DEFAULT_CONFIG = {
  chunk_mode: "character_overlap",
  chunk_size: 700,
  overlap: 120,
  model_name: "sentence-transformers/all-MiniLM-L6-v2",
  top_k: 3,
  candidate_pool: 60,
  answer_model: "openrouter/free",
  retrieval_backend: "faiss",
};

export default function App() {
  const [file, setFile] = useState(null);
  const [uploaded, setUploaded] = useState(false);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [pageCount, setPageCount] = useState(0);
  const [currentPage, setCurrentPage] = useState(1);
  const [pdfJumpKey, setPdfJumpKey] = useState(0);
  const [chatKey, setChatKey] = useState(0);
  const [config, setConfig] = useState(DEFAULT_CONFIG);

  const busy = status !== "";

  /** 上传 PDF — 带上分块和嵌入配置 */
  async function handleUpload() {
    setError("");
    setStatus("Uploading...");
    setUploaded(false);
    try {
      const result = await uploadPDF(file, CHAT_ID, {
        chunk_mode: config.chunk_mode,
        chunk_size: config.chunk_size,
        overlap: config.overlap,
        model_name: config.model_name,
      });
      setPageCount(result.pages);
      setCurrentPage(1);
      setPdfJumpKey((k) => k + 1);
      setChatKey((k) => k + 1);
      setUploaded(true);
      setStatus("");
    } catch (err) {
      setError(err.message);
      setStatus("");
    }
  }

  function handleJumpToPage(page) {
    setCurrentPage(page);
    setPdfJumpKey((k) => k + 1);
  }

  return (
    <div className="app">
      <h1>SmartLearn AI</h1>

      {/* ── 顶部栏：上传 + 配置 ── */}
      <div className="top-bar">
        <form
          className="upload-form"
          onSubmit={(e) => {
            e.preventDefault();
            if (!busy && file) handleUpload();
          }}
        >
          <label>
            PDF 文件：
            <input
              type="file"
              accept=".pdf"
              onChange={(e) => setFile(e.target.files[0] || null)}
            />
          </label>
          <button type="submit" disabled={!file || busy}>
            {uploaded ? "重新上传" : "上传"}
          </button>
          {uploaded && (
            <span className="upload-info">已加载 {pageCount} 页</span>
          )}
        </form>

        <ConfigPanel
          config={config}
          onChange={setConfig}
          disabled={busy}
        />
      </div>

      {/* 状态 & 错误 */}
      {status && <p aria-live="polite" className="status-text">{status}</p>}
      {error && <p role="alert" className="error-text">{error}</p>}

      {/* ── 工作区（上传后显示） ── */}
      {uploaded && (
        <div className="workspace">
          <div className="workspace-left">
            <PdfPreview
              chatId={CHAT_ID}
              targetPage={currentPage}
              jumpKey={pdfJumpKey}
            />
          </div>
          <div className="workspace-right">
            <ChatPanel
              key={chatKey}
              chatId={CHAT_ID}
              enabled={uploaded}
              disabled={busy}
              onBusy={(b) => setStatus(b ? "Asking..." : "")}
              onJumpToPage={handleJumpToPage}
              config={{
                top_k: config.top_k,
                candidate_pool: config.candidate_pool,
                answer_model: config.answer_model,
                retrieval_backend: config.retrieval_backend,
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
