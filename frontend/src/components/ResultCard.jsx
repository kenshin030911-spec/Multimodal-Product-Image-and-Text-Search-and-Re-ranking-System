import { useEffect, useState } from "react";

function formatScore(value) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(3) : "-";
}

function ResultCard({ result, rerankerType, compact = false }) {
  const imageUrl = result.display_image_url || result.image_url;
  const [imageFailed, setImageFailed] = useState(false);

  useEffect(() => {
    setImageFailed(false);
  }, [imageUrl]);

  return (
    <article className={compact ? "result-card result-card-compact" : "result-card"}>
      {imageUrl && !imageFailed ? (
        <img
          src={imageUrl}
          alt={result.title || result.product_id}
          onError={() => setImageFailed(true)}
        />
      ) : (
        <div className="image-placeholder">No Image</div>
      )}

      <div className="result-body">
        <div className="rank-row">
          <span className="rank">#{result.final_rank}</span>
          <span className="recall-rank">召回 #{result.recall_rank}</span>
        </div>

        <h3>{result.title || "Untitled product"}</h3>

        <div className="product-meta">
          <span>ID：{result.product_id}</span>
          <span>{result.article_type || "Unknown type"}</span>
          <span>{result.base_colour || "Unknown colour"}</span>
        </div>

        <dl className="score-list">
          <div>
            <dt>召回分</dt>
            <dd>{formatScore(result.recall_score)}</dd>
          </div>
          <div>
            <dt>重排分</dt>
            <dd>{formatScore(result.rerank_score)}</dd>
          </div>
          <div>
            <dt>新鲜度</dt>
            <dd>{formatScore(result.freshness_score)}</dd>
          </div>
        </dl>
        {rerankerType === "pairwise" ? (
          <p className="score-note compact">
            pairwise rerank_score is an ordering score, not probability.
          </p>
        ) : null}
      </div>
    </article>
  );
}

export default ResultCard;
