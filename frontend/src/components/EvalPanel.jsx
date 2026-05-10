import { TEXT_RERANKER_MODES } from "../api/searchApi.js";
import ResultCard from "./ResultCard.jsx";

const modeLabels = {
  none: "Vector only",
  rule: "Rule reranker",
  trained: "Trained",
  pairwise: "Pairwise"
};

const overlapPairs = [
  ["rule", "pairwise"],
  ["trained", "pairwise"],
  ["none", "pairwise"],
  ["none", "rule"]
];

function EvalPanel({
  query,
  topK,
  selectedModes,
  comparisonResults,
  loading,
  globalError,
  onQueryChange,
  onTopKChange,
  onToggleMode,
  onRun,
  onClear
}) {
  const topKNumber = Number(topK);
  const topKIsValid = topKNumber >= 1 && topKNumber <= 100;
  const successfulModes = selectedModes.filter((mode) => comparisonResults[mode]?.success);
  const failedModes = selectedModes.filter((mode) => comparisonResults[mode]?.error);
  const rankRows = buildRankRows(comparisonResults);
  const overlapStats = buildOverlapStats(comparisonResults, topKNumber);
  const allSelectedFailed = selectedModes.length > 0 && failedModes.length === selectedModes.length;

  return (
    <section className="eval-panel">
      <div className="section-header">
        <h2>Single Query Reranker Comparison</h2>
        <div className="badge-row">
          {loading ? <span className="badge">请求中</span> : null}
          <button className="clear-button" onClick={onClear} type="button">
            清空对比
          </button>
        </div>
      </div>

      <div className="comparison-form">
        <label>
          Query
          <input
            onChange={(event) => onQueryChange(event.target.value)}
            placeholder="black shirt"
            value={query}
          />
        </label>
        <label>
          Top K
          <input
            max="100"
            min="1"
            onChange={(event) => onTopKChange(event.target.value)}
            type="number"
            value={topK}
          />
        </label>
        <fieldset className="reranker-selector">
          <legend>Modes</legend>
          <div className="mode-options">
            {TEXT_RERANKER_MODES.map((mode) => (
              <label
                className={selectedModes.includes(mode) ? "mode-option selected" : "mode-option"}
                key={mode}
              >
                <input
                  checked={selectedModes.includes(mode)}
                  onChange={() => onToggleMode(mode)}
                  type="checkbox"
                />
                <span>{modeLabels[mode]}</span>
                {isExperimentalMode(mode) ? (
                  <span className="experimental-badge">experimental</span>
                ) : null}
              </label>
            ))}
          </div>
        </fieldset>
        <button
          disabled={loading || !query.trim() || selectedModes.length === 0 || !topKIsValid}
          type="button"
          onClick={onRun}
        >
          Run Comparison
        </button>
      </div>

      {globalError ? <p className="error-message">{globalError}</p> : null}
      {allSelectedFailed ? (
        <p className="error-message">所有已选择模式均请求失败，请检查后端服务或模型文件。</p>
      ) : null}

      <div className="summary-grid">
        <SummaryItem label="Query" value={query || "-"} />
        <SummaryItem label="Top K" value={topK || "-"} />
        <SummaryItem label="Enabled modes" value={selectedModes.join(", ") || "-"} />
        <SummaryItem label="Successful modes" value={successfulModes.join(", ") || "-"} />
        <SummaryItem label="Failed modes" value={failedModes.join(", ") || "-"} />
      </div>

      <div className="score-note">
        单 query 对比没有人工 relevance label，因此不计算 Precision / Recall / NDCG / MRR。
        pairwise score 是 ordering score，不是概率；binary trained score 是 probability-like score。
        不同模式分数不要直接比较，只比较排序结果。
      </div>

      <div className="comparison-grid">
        {selectedModes.map((mode) => (
          <ModeColumn
            key={mode}
            mode={mode}
            result={comparisonResults[mode]}
          />
        ))}
      </div>

      <section className="comparison-block">
        <h3>Rank Comparison</h3>
        <div className="comparison-table-wrapper">
          <table className="comparison-table">
            <thead>
              <tr>
                <th>product_id</th>
                <th>title</th>
                <th>none_rank</th>
                <th>rule_rank</th>
                <th>trained_rank</th>
                <th>pairwise_rank</th>
                <th>pairwise_vs_rule_delta</th>
                <th>pairwise_vs_trained_delta</th>
              </tr>
            </thead>
            <tbody>
              {rankRows.length > 0 ? (
                rankRows.map((row) => (
                  <tr key={row.product_id}>
                    <td>{row.product_id}</td>
                    <td>{row.title || "-"}</td>
                    <td>{formatRank(row.none_rank)}</td>
                    <td>{formatRank(row.rule_rank)}</td>
                    <td>{formatRank(row.trained_rank)}</td>
                    <td>{formatRank(row.pairwise_rank)}</td>
                    <td>{formatDelta(row.pairwise_vs_rule_rank_delta)}</td>
                    <td>{formatDelta(row.pairwise_vs_trained_rank_delta)}</td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan="8">暂无可对比结果。</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>

      <section className="comparison-block">
        <h3>Overlap Statistics</h3>
        <div className="overlap-grid">
          {overlapPairs.map(([leftMode, rightMode]) => {
            const key = `${leftMode}_vs_${rightMode}`;
            return (
              <div className="overlap-item" key={key}>
                <span>
                  {leftMode} vs {rightMode}
                </span>
                <strong>{overlapStats[key] || "-"}</strong>
              </div>
            );
          })}
        </div>
      </section>
    </section>
  );
}

function SummaryItem({ label, value }) {
  return (
    <div className="summary-item">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ModeColumn({ mode, result }) {
  const response = result?.response;
  const results = response?.results || [];

  return (
    <section className="mode-column">
      <div className="mode-summary">
        <div className="mode-column-header">
          <h3>{modeLabels[mode]}</h3>
          {isExperimentalMode(mode) ? (
            <span className="experimental-badge">experimental</span>
          ) : null}
        </div>

        {result?.loading ? <p className="loading-state">正在请求...</p> : null}
        {result?.error ? <p className="error-card">{result.error}</p> : null}
        {response?.reranker_message || response?.message ? (
          <p className="status-message mode-message">
            {response.reranker_message || response.message}
          </p>
        ) : null}
      </div>

      {results.length > 0 ? (
        <div className="comparison-result-list">
          {results.map((item) => (
            <ResultCard
              compact
              key={`${mode}-${item.product_id}-${item.final_rank}`}
              rerankerType={mode}
              result={item}
            />
          ))}
        </div>
      ) : !result?.loading && !result?.error ? (
        <p className="empty-state">暂无结果。</p>
      ) : null}
    </section>
  );
}

function buildRankRows(comparisonResults) {
  const rowsByProductId = new Map();

  for (const mode of TEXT_RERANKER_MODES) {
    const response = comparisonResults[mode]?.response;
    if (!comparisonResults[mode]?.success || !response?.results) {
      continue;
    }
    for (const item of response.results) {
      const existing = rowsByProductId.get(item.product_id) || {
        product_id: item.product_id,
        title: item.title,
        none_rank: null,
        rule_rank: null,
        trained_rank: null,
        pairwise_rank: null
      };
      existing.title = existing.title || item.title;
      existing[`${mode}_rank`] = item.final_rank;
      rowsByProductId.set(item.product_id, existing);
    }
  }

  return [...rowsByProductId.values()]
    .map((row) => ({
      ...row,
      pairwise_vs_rule_rank_delta: calculateRankDelta(row.rule_rank, row.pairwise_rank),
      pairwise_vs_trained_rank_delta: calculateRankDelta(
        row.trained_rank,
        row.pairwise_rank
      )
    }))
    .sort((left, right) => sortRankRow(left) - sortRankRow(right));
}

function buildOverlapStats(comparisonResults, topK) {
  return Object.fromEntries(
    overlapPairs.map(([leftMode, rightMode]) => {
      const leftIds = resultIdSet(comparisonResults[leftMode]);
      const rightIds = resultIdSet(comparisonResults[rightMode]);
      const key = `${leftMode}_vs_${rightMode}`;
      if (!leftIds || !rightIds) {
        return [key, "-"];
      }
      const overlapCount = [...leftIds].filter((productId) => rightIds.has(productId)).length;
      return [key, `${overlapCount} / ${topK}`];
    })
  );
}

function resultIdSet(result) {
  if (!result?.success || !result.response?.results?.length) {
    return null;
  }
  return new Set(result.response.results.map((item) => item.product_id));
}

function calculateRankDelta(oldRank, newRank) {
  if (!oldRank || !newRank) {
    return null;
  }
  return oldRank - newRank;
}

function sortRankRow(row) {
  return row.pairwise_rank || row.rule_rank || row.trained_rank || row.none_rank || 9999;
}

function formatRank(value) {
  return value || "-";
}

function formatDelta(value) {
  if (value === null || value === undefined) {
    return "-";
  }
  return value > 0 ? `+${value}` : String(value);
}

function isExperimentalMode(mode) {
  return mode === "trained" || mode === "pairwise";
}

export default EvalPanel;
