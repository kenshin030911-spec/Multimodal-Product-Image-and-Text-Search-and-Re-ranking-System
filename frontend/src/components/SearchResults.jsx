import ResultCard from "./ResultCard.jsx";

function SearchResults({ error, loading, results, status }) {
  const isPlaceholder = status?.placeholder || status?.mock;
  const hasResults = results.length > 0;
  const rerankerType = status?.reranker_type;
  const isExperimental = rerankerType === "trained" || rerankerType === "pairwise";

  return (
    <section className="results-section">
      <div className="section-header">
        <h2>搜索结果</h2>
        <div className="badge-row">
          {loading ? <span className="badge">请求中</span> : null}
          {isPlaceholder ? <span className="badge">placeholder / mock</span> : null}
        </div>
      </div>

      <div className="result-meta">
        {status?.query ? <span>Query：{status.query}</span> : null}
        {status?.query_type ? <span>类型：{status.query_type}</span> : null}
        {status?.top_k ? <span>Top K：{status.top_k}</span> : null}
        {typeof status?.use_rerank === "boolean" ? (
          <span>use_rerank：{status.use_rerank ? "true" : "false"}</span>
        ) : null}
        {rerankerType ? <span>reranker_type：{rerankerType}</span> : null}
        {isExperimental ? <span className="experimental-badge">experimental</span> : null}
      </div>

      {status?.reranker_message || status?.message ? (
        <p className="status-message">{status.reranker_message || status.message}</p>
      ) : null}

      {error ? <p className="error-message">{error}</p> : null}

      {!error && loading && !hasResults ? (
        <p className="loading-state">正在请求后端搜索服务...</p>
      ) : null}

      {!error && !loading && !hasResults ? (
        <p className="empty-state">暂无搜索结果。</p>
      ) : null}

      {hasResults ? (
        <div className={loading ? "result-grid is-loading" : "result-grid"}>
          {results.map((result) => (
            <ResultCard
              key={`${result.product_id}-${result.final_rank}`}
              rerankerType={rerankerType}
              result={result}
            />
          ))}
        </div>
      ) : (
        null
      )}
    </section>
  );
}

export default SearchResults;
