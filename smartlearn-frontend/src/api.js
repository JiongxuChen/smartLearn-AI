const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

/**
 * 上传 PDF 文件到后端
 * @param {File} file
 * @param {string} chatId
 * @param {object} config - { chunk_mode, chunk_size, overlap, model_name }
 */
async function uploadPDF(file, chatId, config = {}) {
  const formData = new FormData();
  formData.append("file", file);

  const params = new URLSearchParams({ chat_id: chatId });
  if (config.chunk_mode) params.set("chunk_mode", config.chunk_mode);
  if (config.chunk_size) params.set("chunk_size", config.chunk_size);
  if (config.overlap != null) params.set("overlap", config.overlap);
  if (config.model_name) params.set("model_name", config.model_name);

  const res = await fetch(`${API}/upload?${params}`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail || `上传失败（${res.status}）`);
  }

  return res.json();
}

/**
 * 向已上传的 PDF 提问题
 * @param {string} message
 * @param {string} chatId
 * @param {object} config - { top_k, candidate_pool, answer_model, retrieval_backend }
 */
async function askQuestion(message, chatId, config = {}) {
  const body = {
    message,
    ...(config.top_k != null && { top_k: config.top_k }),
    ...(config.candidate_pool != null && { candidate_pool: config.candidate_pool }),
    ...(config.answer_model && { answer_model: config.answer_model }),
    ...(config.retrieval_backend && { retrieval_backend: config.retrieval_backend }),
  };

  const res = await fetch(`${API}/chat?chat_id=${chatId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const errBody = await res.json().catch(() => null);
    throw new Error(errBody?.detail || `对话失败（${res.status}）`);
  }

  return res.json();
}

function getPdfUrl(chatId) {
  return `${API}/documents/${chatId}/file`;
}

async function resetChat(chatId) {
  const res = await fetch(`${API}/chat/reset?chat_id=${chatId}`, {
    method: "POST",
  });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail || `重置失败（${res.status}）`);
  }
  return res.json();
}

export { API, uploadPDF, askQuestion, getPdfUrl, resetChat };

