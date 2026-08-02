const API = import.meta.env.VITE_API_URL || "http://localhost:8000";
const CHAT_ID = "day2-demo";

/**
 * 上传 PDF 文件到后端
 * @param {File} file - 用户选择的 PDF 文件
 * @returns {Promise<{chat_id: string, pages: Array<{page: number, text: string}>}>}
 */
async function uploadPDF(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API}/upload?chat_id=${CHAT_ID}`, {
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
 * @param {string} message - 用户问题
 * @returns {Promise<{chat_id: string, answer: string}>}
 */
async function askQuestion(message) {
  const res = await fetch(`${API}/chat?chat_id=${CHAT_ID}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });

  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail || `对话失败（${res.status}）`);
  }

  return res.json();
}

export { API, CHAT_ID, uploadPDF, askQuestion };
