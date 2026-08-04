import { useState, useEffect } from "react";
import { askQuestion, resetChat } from "./api";

export default function ChatPanel({
  chatId,
  enabled,
  onBusy,
  disabled,
  onJumpToPage,
  config = {},
}) {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // 新上传后清空对话
  useEffect(() => {
    setMessages([]);
    setError("");
  }, [enabled]);

  async function handleReset() {
    try {
      await resetChat(chatId);
      setMessages([]);
      setError("");
    } catch (err) {
      setError(err.message);
    }
  }

  async function handleSubmit(e) {
    e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || loading || disabled) return;

    setInput("");
    setError("");
    setLoading(true);
    onBusy?.(true);

    // 先追加用户消息
    const userMsg = { role: "user", question: trimmed };
    setMessages((prev) => [...prev, userMsg]);

    try {
      const result = await askQuestion(trimmed, chatId, config);
      const assistantMsg = {
        role: "assistant",
        answer: result.answer,
        citations: result.citations || [],
        sources: result.sources || [],
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
      onBusy?.(false);
    }
  }

  return (
    <div className="chat-panel">
      {/* 消息列表 */}
      <div className="chat-messages">
        {messages.length > 0 && (
          <div className="chat-reset-bar">
            <button
              className="chat-reset-btn"
              onClick={handleReset}
              disabled={loading}
              type="button"
            >
              &#8635; 新对话
            </button>
          </div>
        )}
        {messages.length === 0 && !loading && (
          <p className="chat-empty">
            {enabled ? "输入问题开始对话" : "上传 PDF 后即可开始提问"}
          </p>
        )}
        {messages.map((msg, i) =>
          msg.role === "user" ? (
            <div key={i} className="chat-question">
              <span className="chat-role">你</span>
              <p>{msg.question}</p>
            </div>
          ) : (
            <div key={i} className="chat-answer">
              <span className="chat-role">助手</span>
              <div className="chat-answer-body">
                {msg.citations.length > 0 && (
                  <div className="chat-citations">
                    {msg.citations.map((page) => (
                      <button
                        key={page}
                        className="citation-chip"
                        onClick={() => onJumpToPage(page)}
                        type="button"
                      >
                        Page {page}
                      </button>
                    ))}
                  </div>
                )}
                <div className="chat-answer-text">{msg.answer}</div>
              </div>
            </div>
          ),
        )}
        {loading && <p className="chat-loading">思考中…</p>}
        {error && <p className="chat-error">{error}</p>}
      </div>

      {/* 输入区 */}
      <form className="chat-input-form" onSubmit={handleSubmit}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="基于 PDF 内容提问…（支持多轮对话）"
          disabled={loading || disabled}
          rows={2}
        />
        <button type="submit" disabled={!input.trim() || loading || disabled}>
          提问
        </button>
      </form>
    </div>
  );
}
