import { useState } from "react";
import { uploadPDF, askQuestion } from "./api";

/** 从 LLM 回答中提取 [Page N] 和 [Pages X-Y] 引用，去重排序 */
function extractCitations(text) {
  const re = /\[Pages?\s*(\d+)(?:\s*[-–]\s*(\d+))?\]/g;
  const pages = new Set();
  let m;
  while ((m = re.exec(text)) !== null) {
    const start = parseInt(m[1], 10);
    const end = m[2] ? parseInt(m[2], 10) : start;
    for (let p = start; p <= end; p++) pages.add(p);
  }
  return [...pages].sort((a, b) => a - b);
}

export default function App() {
  const [file, setFile] = useState(null);       // 用户选择的 PDF 文件
  const [upload, setUpload] = useState(null);    // 上传结果 { chat_id, pages }
  const [message, setMessage] = useState("");    // 用户输入的问题
  const [answer, setAnswer] = useState(null);    // 回答 { text, citations }
  const [status, setStatus] = useState("");      // "Uploading..." / "Asking..." / ""
  const [error, setError] = useState("");        // 错误信息

  const busy = status !== "";

  /** 上传 PDF */
  async function handleUpload() {
    setError("");
    setStatus("Uploading...");
    setUpload(null);
    setAnswer(null);
    try {
      const result = await uploadPDF(file);
      setUpload(result);
      setStatus("");
    } catch (err) {
      setError(err.message);
      setStatus("");
    }
  }

  /** 向 PDF 提问 */
  async function handleAsk() {
    setError("");
    setStatus("Asking...");
    try {
      const result = await askQuestion(message.trim());
      setAnswer({
        text: result.answer,
        citations: extractCitations(result.answer),
      });
      setStatus("");
    } catch (err) {
      setError(err.message);
      setStatus("");
    }
  }

  return (
    <>
      <h1>SmartLearn AI</h1>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (!busy && file) handleUpload();
        }}
      >
        {/* ── PDF 上传区 ── */}
        <label>
          PDF 文件：
          <input
            type="file"
            accept=".pdf"
            onChange={(e) => setFile(e.target.files[0] || null)}
          />
        </label>
        <button type="submit" disabled={!file || busy}>
          上传
        </button>
      </form>

      {/* ── 状态提示 ── */}
      {status && <p aria-live="polite">{status}</p>}

      {/* ── 错误信息 ── */}
      {error && (
        <p role="alert" style={{ color: "red" }}>
          {error}
        </p>
      )}

      {/* ── 上传结果 ── */}
      {upload && (
        <p>
          已上传：{upload.pages.length} 页
        </p>
      )}

      {/* ── 问答区 ── */}
      {upload && (
        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (!busy && message.trim()) handleAsk();
          }}
        >
          <label>
            你的问题：
            <textarea
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              placeholder="基于 PDF 内容提问…"
            />
          </label>
          <button type="submit" disabled={!message.trim() || busy}>
            提问
          </button>
        </form>
      )}

      {/* ── 回答结果 ── */}
      {answer && (
        <section>
          {/* 页码引用 chips */}
          {answer.citations.length > 0 && (
            <div>
              {answer.citations.map((page) => (
                <span key={page} className="citation-chip">
                  Page {page}
                </span>
              ))}
            </div>
          )}

          {/* 回答正文 */}
          <div style={{ whiteSpace: "pre-wrap" }}>{answer.text}</div>
        </section>
      )}
    </>
  );
}
