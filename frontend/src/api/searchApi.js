const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
const USE_MOCK_API = import.meta.env.VITE_USE_MOCK_API === "true";
export const TEXT_RERANKER_MODES = ["none", "rule", "trained", "pairwise"];
export const IMAGE_RERANKER_MODES = ["none", "rule"];

const mockResults = [
  {
    product_id: "MOCK-001",
    title: "Mock Black Summer Dress",
    image_url: null,
    image_path: null,
    article_type: "Dresses",
    base_colour: "Black",
    recall_rank: 1,
    recall_score: 0.78,
    rerank_score: 0.84,
    freshness_score: 0.62,
    final_rank: 1
  },
  {
    product_id: "MOCK-002",
    title: "Mock White Running Sneakers",
    image_url: null,
    image_path: null,
    article_type: "Sports Shoes",
    base_colour: "White",
    recall_rank: 2,
    recall_score: 0.72,
    rerank_score: 0.79,
    freshness_score: 0.68,
    final_rank: 2
  },
  {
    product_id: "MOCK-003",
    title: "Mock Black Cotton Shirt",
    image_url: null,
    image_path: null,
    article_type: "Shirts",
    base_colour: "Black",
    recall_rank: 3,
    recall_score: 0.69,
    rerank_score: 0.76,
    freshness_score: 0.58,
    final_rank: 3
  },
  {
    product_id: "MOCK-004",
    title: "Mock Navy Casual Shirt",
    image_url: null,
    image_path: null,
    article_type: "Shirts",
    base_colour: "Navy Blue",
    recall_rank: 4,
    recall_score: 0.65,
    rerank_score: 0.71,
    freshness_score: 0.74,
    final_rank: 4
  }
];

const mockModeOrders = {
  none: ["MOCK-001", "MOCK-002", "MOCK-003", "MOCK-004"],
  rule: ["MOCK-003", "MOCK-001", "MOCK-004", "MOCK-002"],
  trained: ["MOCK-001", "MOCK-003", "MOCK-002", "MOCK-004"],
  pairwise: ["MOCK-003", "MOCK-004", "MOCK-001", "MOCK-002"]
};

const mockMessages = {
  none: "前端 mock 响应：基础向量检索完成，reranker 未启用。",
  rule: "前端 mock 响应：基础向量检索完成，已启用规则 reranker baseline。",
  trained: "前端 mock 响应：基础向量检索完成，已启用 experimental trained reranker。",
  pairwise:
    "前端 mock 响应：基础向量检索完成，已启用 experimental pairwise reranker。pairwise score is an ordering score, not calibrated probability."
};

function buildMockSearchResponse(queryType, query, topK, rerankerType = "rule") {
  const normalizedMode = normalizeRerankerType(rerankerType);
  const useRerank = normalizedMode !== "none";
  const order = mockModeOrders[normalizedMode] || mockModeOrders.rule;
  const byProductId = new Map(mockResults.map((result) => [result.product_id, result]));
  const results = order
    .map((productId) => byProductId.get(productId))
    .filter(Boolean)
    .slice(0, topK)
    .map((result, index) => ({
      ...result,
      final_rank: index + 1,
      rerank_score: mockScoreForMode(result, normalizedMode, index)
    }));

  // mock: true 用来提醒这不是后端真实检索结果。
  return {
    query_type: queryType,
    query,
    top_k: topK,
    use_rerank: useRerank,
    reranker_type: normalizedMode,
    reranker_message: mockMessages[normalizedMode],
    results,
    placeholder: true,
    mock: true,
    message: mockMessages[normalizedMode]
  };
}

export async function searchText({
  query,
  topK = 20,
  rerankerType = null,
  useRerank = true
}) {
  const normalizedMode = normalizeRerankerType(rerankerType, useRerank);
  const normalizedUseRerank = normalizedMode !== "none";

  if (USE_MOCK_API) {
    return normalizeSearchResponse(buildMockSearchResponse("text", query, topK, normalizedMode));
  }

  return requestJson(
    `${API_BASE_URL}/api/search/text`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        top_k: topK,
        reranker_type: normalizedMode,
        use_rerank: normalizedUseRerank
      })
    },
    { normalizeSearch: true }
  );
}

export async function searchImage({
  file,
  topK = 20,
  rerankerType = null,
  useRerank = true
}) {
  const normalizedMode = normalizeImageRerankerType(rerankerType, useRerank);
  const normalizedUseRerank = normalizedMode !== "none";

  if (USE_MOCK_API) {
    return normalizeSearchResponse(
      buildMockSearchResponse(
        "image",
        file?.name || "uploaded-image",
        topK,
        normalizedMode
      )
    );
  }

  const formData = new FormData();
  formData.append("file", file);
  formData.append("top_k", String(topK));
  formData.append("reranker_type", normalizedMode);
  formData.append("use_rerank", String(normalizedUseRerank));

  return requestJson(
    `${API_BASE_URL}/api/search/image`,
    {
      method: "POST",
      body: formData
    },
    { normalizeSearch: true }
  );
}

export async function compareTextSearchModes({
  query,
  topK = 10,
  modes = TEXT_RERANKER_MODES
}) {
  const uniqueModes = [...new Set(modes.filter((mode) => TEXT_RERANKER_MODES.includes(mode)))];
  const settled = await Promise.allSettled(
    uniqueModes.map((mode) => searchText({ query, topK, rerankerType: mode }))
  );

  return uniqueModes.reduce((accumulator, mode, index) => {
    const result = settled[index];
    if (result.status === "fulfilled") {
      accumulator[mode] = {
        status: "success",
        response: result.value
      };
    } else {
      accumulator[mode] = {
        status: "error",
        error: result.reason?.message || "请求失败。"
      };
    }
    return accumulator;
  }, {});
}

export async function getEvaluationSummary() {
  if (USE_MOCK_API) {
    return {
      metrics: {
        precision_at_10_before_rerank: 0,
        precision_at_10_after_rerank: 0,
        recall_at_10_before_rerank: 0,
        recall_at_10_after_rerank: 0,
        ndcg_at_10_before_rerank: 0,
        ndcg_at_10_after_rerank: 0
      },
      placeholder: true,
      mock: true,
      message: "前端 mock 评估摘要：后续会展示真实评估结果。"
    };
  }

  return requestJson(`${API_BASE_URL}/api/evaluation/summary`);
}

export function resolveApiUrl(pathOrUrl) {
  if (!pathOrUrl) {
    return null;
  }

  if (pathOrUrl.startsWith("http://") || pathOrUrl.startsWith("https://")) {
    return pathOrUrl;
  }

  if (pathOrUrl.startsWith("/")) {
    return `${API_BASE_URL}${pathOrUrl}`;
  }

  return pathOrUrl;
}

export function normalizeSearchResponse(response) {
  return {
    ...response,
    reranker_type: response.reranker_type || (response.use_rerank ? "rule" : "none"),
    reranker_message: response.reranker_message || response.message,
    results: (response.results || []).map((result) => ({
      ...result,
      display_image_url: resolveApiUrl(result.image_url)
    }))
  };
}

async function requestJson(url, options = {}, { normalizeSearch = false } = {}) {
  let response;
  try {
    response = await fetch(url, options);
  } catch (error) {
    throw new Error("后端可能未启动或无法连接。");
  }

  const body = await readJsonBody(response);
  if (!response.ok) {
    throw new Error(extractErrorMessage(body) || `请求失败：HTTP ${response.status}`);
  }

  return normalizeSearch ? normalizeSearchResponse(body) : body;
}

async function readJsonBody(response) {
  try {
    return await response.json();
  } catch (error) {
    return null;
  }
}

function extractErrorMessage(body) {
  if (!body) {
    return "";
  }
  if (typeof body.detail === "string") {
    return body.detail;
  }
  if (Array.isArray(body.detail)) {
    return body.detail
      .map((item) => item.msg || item.message || JSON.stringify(item))
      .join("；");
  }
  if (body.detail) {
    return JSON.stringify(body.detail);
  }
  if (typeof body.message === "string") {
    return body.message;
  }
  return "";
}

function normalizeRerankerType(rerankerType, useRerank = true) {
  if (TEXT_RERANKER_MODES.includes(rerankerType)) {
    return rerankerType;
  }
  return useRerank ? "rule" : "none";
}

function normalizeImageRerankerType(rerankerType, useRerank = true) {
  const mode = normalizeRerankerType(rerankerType, useRerank);
  if (!IMAGE_RERANKER_MODES.includes(mode)) {
    throw new Error("trained / pairwise currently support text search only.");
  }
  return mode;
}

function mockScoreForMode(result, mode, index) {
  if (mode === "none") {
    return result.recall_score;
  }
  if (mode === "pairwise") {
    return 1.5 - index * 0.18 + result.freshness_score * 0.1;
  }
  if (mode === "trained") {
    return 0.86 - index * 0.06;
  }
  return 0.9 - index * 0.05;
}
