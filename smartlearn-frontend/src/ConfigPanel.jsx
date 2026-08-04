const CHUNK_MODES = [
  { value: "character_overlap", label: "Character Overlap" },
  { value: "character", label: "Character Fixed" },
  { value: "paragraph", label: "Paragraph" },
  { value: "langchain_recursive", label: "Recursive Split" },
];

const BACKENDS = [
  { value: "faiss", label: "FAISS" },
  { value: "chroma", label: "Chroma" },
];

export default function ConfigPanel({ config, onChange, disabled }) {
  function update(key, value) {
    onChange({ ...config, [key]: value });
  }

  return (
    <details className="config-panel">
      <summary className="config-toggle">&#9881; Config</summary>
      <div className="config-grid">
        {/* 分块 */}
        <label className="config-field">
          Chunk Mode
          <select
            value={config.chunk_mode}
            onChange={(e) => update("chunk_mode", e.target.value)}
            disabled={disabled}
          >
            {CHUNK_MODES.map((m) => (
              <option key={m.value} value={m.value}>{m.label}</option>
            ))}
          </select>
        </label>
        <label className="config-field">
          Chunk Size
          <input
            type="number"
            value={config.chunk_size}
            onChange={(e) => update("chunk_size", Number(e.target.value))}
            disabled={disabled}
            min={100} max={2000} step={50}
          />
        </label>

        {(config.chunk_mode === "character_overlap" || config.chunk_mode === "langchain_recursive") && (
          <label className="config-field">
            Overlap
            <input
              type="number"
              value={config.overlap}
              onChange={(e) => update("overlap", Number(e.target.value))}
              disabled={disabled}
              min={0} max={500} step={10}
            />
          </label>
        )}

        {/* 检索 */}
        <label className="config-field">
          Retrieval
          <select
            value={config.retrieval_backend}
            onChange={(e) => update("retrieval_backend", e.target.value)}
          >
            {BACKENDS.map((b) => (
              <option key={b.value} value={b.value}>{b.label}</option>
            ))}
          </select>
        </label>
        <label className="config-field">
          Top-K
          <input
            type="number"
            value={config.top_k}
            onChange={(e) => update("top_k", Number(e.target.value))}
            min={1} max={10}
          />
        </label>

        {/* LLM */}
        <label className="config-field config-field--wide">
          LLM Model
          <input
            type="text"
            value={config.answer_model}
            onChange={(e) => update("answer_model", e.target.value)}
            placeholder="openrouter/free"
          />
        </label>
      </div>
      <p className="config-hint">
        Chunk settings apply on upload. Retrieval &amp; LLM apply instantly.
      </p>
    </details>
  );
}
