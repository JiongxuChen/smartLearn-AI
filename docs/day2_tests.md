# Day 2 E2E Tests

测试日期：2026-08-02
CHAT_ID：`day2-demo`
后端：`http://localhost:8080`
前端：Vite React（代码级验证）

---

## Test 1: Upload 按钮禁用逻辑

**预期：** 未选文件时 Upload 按钮 `disabled`，选文件后启用。

**验证方式：** 代码审查 `App.jsx:79`

```jsx
<button type="submit" disabled={!file || busy}>
```

| 状态 | `file` | `busy` | disabled 计算结果 |
|---|---|---|---|
| 初始加载 | `null` | `false` | `true` ✅ |
| 用户选了文件 | `File` 对象 | `false` | `false` ✅ |
| 上传中 | `File` 对象 | `true`（"Uploading..."） | `true` ✅ |

**结果：PASS**

---

## Test 2: 正常 PDF 上传

**预期：** `POST /upload?chat_id=day2-demo` 返回 200，解析出页数和文本。

**命令：**
```bash
curl -X POST "http://127.0.0.1:8080/upload?chat_id=day2-demo" \
  -F "file=@test_files/sample.pdf"
```

**实际响应：**
```json
{
  "chat_id": "day2-demo",
  "pages": [
    {"page": 1, "text": "Attention Is All You Need\nAshish Vaswani…"},
    {"page": 2, "text": "Recurrent models typically…"},
    …
  ]
}
```

| 指标 | 值 |
|---|---|
| HTTP 状态码 | **200** |
| 页数 | **11** |
| 总字符数 | **32,643** |

**结果：PASS**

---

## Test 3: Ask 按钮禁用逻辑

**预期：** 上传前不显示问答区；上传后 blank 消息时按钮 `disabled`。

**验证方式：** 代码审查 `App.jsx:102, 117`

```jsx
{upload && (                           // 仅上传后渲染问答区
  <form …>
    <button disabled={!message.trim() || busy}>
```

| 状态 | `upload` | `message` | `busy` | disabled |
|---|---|---|---|---|
| 上传前 | `null` | `""` | `false` | 问答区不渲染 ✅ |
| 已上传，空白消息 | `{pages:[…]}` | `""` | `false` | `true` ✅ |
| 已上传，有消息 | `{pages:[…]}` | `"hello"` | `false` | `false` ✅ |
| 提问中 | `{pages:[…]}` | `"hello"` | `true`（"Asking..."） | `true` ✅ |

**结果：PASS**

---

## Test 4: 已知问题返回答案 + Page chips

**预期：** 对 PDF 中存在的事实提问，返回带 `[Page N]` 引用的答案，前端提取为 Page chips。

**命令：**
```bash
curl -X POST "http://127.0.0.1:8080/chat?chat_id=day2-demo" \
  -H "Content-Type: application/json" \
  -d '{"message":"How many encoder layers in Transformer?"}'
```

**实际响应：**
```json
{
  "chat_id": "day2-demo",
  "answer": "The Transformer's encoder consists of a stack of **6 identical layers**. \n[Page 3]"
}
```

| 验证点 | 结果 |
|---|---|
| HTTP 状态码 | **200** |
| 答案包含事实 | ✅ "6 identical layers" |
| 包含 `[Page N]` 引用 | ✅ `[Page 3]` |
| `extractCitations("[Page 3]")` | `[3]` → 渲染 chip `"Page 3"` |

`extractCitations` 正则测试：
```js
// App.jsx:5-14
extractCitations("…6 layers. [Page 3] The decoder also…")  
// → [3]

extractCitations("…detailed in [Pages 3-5] of the paper…")  
// → [3, 4, 5]
```

**结果：PASS**

---

## Test 5: PDF 中不存在的信息不编造

**预期：** 问 PDF 完全没有的内容，LLM 拒绝编造，不会出现假 `[Page X]`。

**命令：**
```bash
curl -X POST "http://127.0.0.1:8080/chat?chat_id=day2-demo" \
  -H "Content-Type: application/json" \
  -d '{"message":"What is the capital of France?"}'
```

**实际响应：**
```json
{
  "chat_id": "day2-demo",
  "answer": "The document does not provide enough information to answer this question."
}
```

| 验证点 | 结果 |
|---|---|
| HTTP 状态码 | **200** |
| 拒绝编造 | ✅ 明确说文档不含此信息 |
| 无假页码 | ✅ 没有 `[Page X]` 模式出现 |

**结果：PASS**

---

## Test 6: 后端宕机 → 前端可见错误

**预期：** 后端不运行时，`fetch()` 失败，`api.js` 捕获并抛出可读错误，前端 `<p role="alert">` 展示。

**验证方式：** 停止后端 → curl 请求

```bash
# 后端已被 kill
curl -X POST "http://127.0.0.1:8080/chat?chat_id=day2-demo" \
  -H "Content-Type: application/json" \
  -d '{"message":"What is the Transformer?"}'
```

**实际结果：**
```
curl: (7) Failed to connect to 127.0.0.1 port 8080
HTTP_STATUS: 000
```

**前端行为（`api.js:18-21` 处理链）：**
```
fetch() reject
  → api.js 中 res.ok 走不到（res 不存在）
    → fetch 抛 TypeError: Failed to fetch
      → handleAsk catch(err) → setError("TypeError: Failed to fetch")
        → <p role="alert"> 红色显示错误信息
```

| 验证点 | 结果 |
|---|---|
| 后端不可达 | ✅ `Failed to connect` |
| HTTP 状态 | **000**（无响应） |
| 前端 `role="alert"` 元素显示错误 | ✅ 代码路径确认 |

**结果：PASS**

---

## Test 7: 重启后端必须重新上传

**预期：** 后端用内存存储 `chat_store`，重启后清空。已上传的 `chat_id` 返回 404，重新上传后恢复可用。

### 7a：重启后不重新上传，直接提问

```bash
curl -X POST "http://127.0.0.1:8080/chat?chat_id=day2-demo" \
  -H "Content-Type: application/json" \
  -d '{"message":"How many encoder layers?"}'
```

**实际响应：**
```json
{"detail": "chat_id 未找到，请先上传 PDF"}
```

| HTTP 状态码 | **404** |

### 7b：重新上传后再提问

```bash
# 重新上传
curl -X POST "http://127.0.0.1:8080/upload?chat_id=day2-demo" \
  -F "file=@test_files/sample.pdf"

# 再次提问
curl -X POST "http://127.0.0.1:8080/chat?chat_id=day2-demo" \
  -H "Content-Type: application/json" \
  -d '{"message":"How many encoder layers?"}'
```

**实际响应：**
```json
{"chat_id": "day2-demo", "answer": "The Transformer encoder consists of a stack of 6 identical layers. [Page 2]"}
```

| HTTP 状态码 | **200** |

| 步骤 | 预期 | 实际 | 状态 |
|---|---|---|---|
| 重启后不重新上传 | 404 | 404 `"chat_id 未找到，请先上传 PDF"` | ✅ |
| 重新上传 | 200 | 200，11 页 | ✅ |
| 上传后再提问 | 200 + 答案 | 200，`"6 identical layers. [Page 2]"` | ✅ |

**结果：PASS**

---

## 边界 PDF 类型测试

| 文件 | 预期 HTTP | 实际 HTTP | `detail` | 状态 |
|---|---|---|---|---|
| `sample.pdf`（正常） | 200 | 200 | — | ✅ |
| `empty.pdf`（空文件） | 400 | 400 | `"文件为空，请上传有效的 PDF 文件"` | ✅ |
| `sample-scan.pdf`（扫描件） | 422 | 422 | `"该 PDF 为扫描件或图片，无文字层。OCR 功能暂不支持"` | ✅ |
| `large_file.pdf`（44 页） | 400 | 400 | `"PDF 最多允许 30 页，当前为 44 页"` | ✅ |
| 非 PDF 文件 | 400 | 400 | `"无法识别的文件格式，请上传 PDF 文件"` | ✅ |

---

## 总结

| Test | 描述 | 结果 |
|---|---|---|
| 1 | Upload 按钮禁用逻辑 | ✅ PASS |
| 2 | 正常 PDF 上传 | ✅ PASS |
| 3 | Ask 按钮禁用逻辑 | ✅ PASS |
| 4 | 已知问题 → 答案 + Page chips | ✅ PASS |
| 5 | PDF 不含的信息不编造 | ✅ PASS |
| 6 | 后端宕机 → 前端可见错误 | ✅ PASS |
| 7 | 重启后端 → 需重新上传 | ✅ PASS |
| — | 5 种 PDF 边界类型 | ✅ ALL PASS |

**7/7 全部通过。**
