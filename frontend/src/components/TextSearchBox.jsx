import { TEXT_RERANKER_MODES } from "../api/searchApi.js";

const modeLabels = {
  none: "Vector only",
  rule: "Rule reranker default",
  trained: "Trained reranker",
  pairwise: "Pairwise reranker"
};

function TextSearchBox({
  onSearch,
  loading,
  query,
  topK,
  rerankerType,
  onQueryChange,
  onTopKChange,
  onRerankerTypeChange
}) {
  const currentMode = rerankerType || "rule";
  const topKNumber = Number(topK);
  const topKIsValid = topKNumber >= 1 && topKNumber <= 100;

  function handleSubmit(event) {
    event.preventDefault();
    if (!topKIsValid) {
      return;
    }
    onSearch({ query, topK: topKNumber, rerankerType: currentMode });
  }

  return (
    <form className="search-box" onSubmit={handleSubmit}>
      <label>
        文本查询
        <input
          value={query}
          onChange={(event) => onQueryChange(event.target.value)}
          placeholder="black summer dress"
        />
      </label>

      <div className="control-row">
        <label>
          Top K
          <input
            min="1"
            max="100"
            type="number"
            value={topK}
            onChange={(event) => onTopKChange(event.target.value)}
          />
        </label>
      </div>

      <fieldset className="reranker-selector">
        <legend>Reranker</legend>
        <div className="mode-options">
          {TEXT_RERANKER_MODES.map((mode) => (
            <label
              className={currentMode === mode ? "mode-option selected" : "mode-option"}
              key={mode}
            >
              <input
                checked={currentMode === mode}
                name="text-reranker-type"
                onChange={() => onRerankerTypeChange(mode)}
                type="radio"
              />
              <span>{modeLabels[mode]}</span>
              {mode === "trained" || mode === "pairwise" ? (
                <span className="experimental-badge">experimental</span>
              ) : null}
            </label>
          ))}
        </div>
      </fieldset>

      <button type="submit" disabled={loading || !query.trim() || !topKIsValid}>
        文本搜索
      </button>
    </form>
  );
}

export default TextSearchBox;
