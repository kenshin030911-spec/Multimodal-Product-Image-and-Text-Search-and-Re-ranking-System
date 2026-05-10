import { useState } from "react";
import ImageUploadBox from "../components/ImageUploadBox.jsx";
import SearchResults from "../components/SearchResults.jsx";
import TextSearchBox from "../components/TextSearchBox.jsx";
import { IMAGE_RERANKER_MODES, TEXT_RERANKER_MODES, searchImage, searchText } from "../api/searchApi.js";

const SEARCH_STORAGE_KEY = "fashion_search_state_v1";

const defaultStatus = {
  placeholder: true,
  message: "请选择文本搜索或图片搜索。",
  query_type: null,
  query: null,
  top_k: null,
  use_rerank: null,
  reranker_type: null,
  reranker_message: null
};

function SearchPage() {
  const restoredState = loadSearchState();
  const [textQuery, setTextQuery] = useState(restoredState?.textQuery || "black summer dress");
  const [topK, setTopK] = useState(String(restoredState?.topK || 5));
  const [textRerankerType, setTextRerankerType] = useState(
    normalizeTextMode(restoredState?.textRerankerType || restoredState?.rerankerType)
  );
  const [imageRerankerType, setImageRerankerType] = useState(
    normalizeImageMode(restoredState?.imageRerankerType || restoredState?.rerankerType)
  );
  const [imageFileName, setImageFileName] = useState(restoredState?.imageFileName || "");
  const [results, setResults] = useState(restoredState?.results || []);
  const [status, setStatus] = useState(restoredState?.status || defaultStatus);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(restoredState?.error || "");
  const [clearToken, setClearToken] = useState(0);

  async function runSearch(searcher, payload, searchType) {
    if (loading) {
      return;
    }
    setLoading(true);
    setError("");
    try {
      const response = await searcher(payload);
      const nextResults = response.results || [];
      const nextStatus = buildStatusFromResponse(response, payload, searchType);
      setResults(nextResults);
      setStatus(nextStatus);
      persistSearchState({
        searchType,
        textQuery,
        imageFileName: payload.imageFileName || imageFileName,
        topK: payload.topK,
        rerankerType: nextStatus.reranker_type,
        textRerankerType,
        imageRerankerType,
        useRerank: nextStatus.use_rerank,
        lastResponse: response,
        results: nextResults,
        status: nextStatus,
        error: "",
        updatedAt: new Date().toISOString()
      });
    } catch (error) {
      const message = error.message || "搜索请求失败。";
      const nextStatus = buildStatusFromError(payload, searchType);
      setResults([]);
      setError(message);
      setStatus(nextStatus);
      persistSearchState({
        searchType,
        textQuery,
        imageFileName: payload.imageFileName || imageFileName,
        topK: payload.topK,
        rerankerType: payload.rerankerType,
        textRerankerType,
        imageRerankerType,
        useRerank: payload.rerankerType !== "none",
        lastResponse: null,
        results: [],
        status: nextStatus,
        error: message,
        updatedAt: new Date().toISOString()
      });
    } finally {
      setLoading(false);
    }
  }

  function clearSearchState() {
    sessionStorage.removeItem(SEARCH_STORAGE_KEY);
    setTextQuery("black summer dress");
    setTopK("5");
    setTextRerankerType("rule");
    setImageRerankerType("rule");
    setImageFileName("");
    setResults([]);
    setStatus(defaultStatus);
    setError("");
    setClearToken((value) => value + 1);
  }

  return (
    <div className="page-layout">
      <section className="tool-panel">
        <div className="section-header">
          <h2>检索入口</h2>
          <div className="badge-row">
            {loading ? <span className="badge">请求中</span> : null}
            <button className="clear-button" onClick={clearSearchState} type="button">
              清空结果
            </button>
          </div>
        </div>
        <div className="search-grid">
          <TextSearchBox
            loading={loading}
            onQueryChange={setTextQuery}
            onRerankerTypeChange={setTextRerankerType}
            onSearch={(payload) => runSearch(searchText, payload, "text")}
            onTopKChange={setTopK}
            query={textQuery}
            rerankerType={textRerankerType}
            topK={topK}
          />
          <ImageUploadBox
            clearToken={clearToken}
            imageFileName={imageFileName}
            loading={loading}
            onImageFileNameChange={setImageFileName}
            onRerankerTypeChange={setImageRerankerType}
            onSearch={(payload) => runSearch(searchImage, payload, "image")}
            onTopKChange={setTopK}
            rerankerType={imageRerankerType}
            topK={topK}
          />
        </div>
      </section>

      <SearchResults
        error={error}
        loading={loading}
        results={results}
        status={status}
      />
    </div>
  );
}

function buildStatusFromResponse(response, payload, searchType) {
  return {
    placeholder: Boolean(response.placeholder),
    mock: response.mock,
    message: response.message,
    query_type: response.query_type || searchType,
    query: response.query || payload.query || payload.imageFileName || null,
    top_k: response.top_k || payload.topK,
    use_rerank: response.use_rerank,
    reranker_type: response.reranker_type || payload.rerankerType,
    reranker_message: response.reranker_message || response.message
  };
}

function buildStatusFromError(payload, searchType) {
  return {
    ...defaultStatus,
    placeholder: true,
    query_type: searchType,
    query: payload.query || payload.imageFileName || null,
    top_k: payload.topK,
    use_rerank: payload.rerankerType !== "none",
    reranker_type: payload.rerankerType
  };
}

function persistSearchState(state) {
  try {
    sessionStorage.setItem(SEARCH_STORAGE_KEY, JSON.stringify(state));
  } catch (error) {
    // sessionStorage may be unavailable in strict browser settings.
  }
}

function loadSearchState() {
  try {
    const rawValue = sessionStorage.getItem(SEARCH_STORAGE_KEY);
    if (!rawValue) {
      return null;
    }
    return JSON.parse(rawValue);
  } catch (error) {
    sessionStorage.removeItem(SEARCH_STORAGE_KEY);
    return null;
  }
}

function normalizeTextMode(mode) {
  return TEXT_RERANKER_MODES.includes(mode) ? mode : "rule";
}

function normalizeImageMode(mode) {
  return IMAGE_RERANKER_MODES.includes(mode) ? mode : "rule";
}

export default SearchPage;
