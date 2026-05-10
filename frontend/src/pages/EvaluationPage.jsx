import { useState } from "react";
import EvalPanel from "../components/EvalPanel.jsx";
import { TEXT_RERANKER_MODES, compareTextSearchModes } from "../api/searchApi.js";

const EVAL_STORAGE_KEY = "fashion_eval_compare_state_v1";

function EvaluationPage() {
  const restoredState = loadEvalState();
  const [query, setQuery] = useState(restoredState?.query || "black shirt");
  const [topK, setTopK] = useState(String(restoredState?.topK || 10));
  const [selectedModes, setSelectedModes] = useState(
    restoredState?.selectedModes?.length ? restoredState.selectedModes : TEXT_RERANKER_MODES
  );
  const [comparisonResults, setComparisonResults] = useState(
    restoredState?.comparisonResponses || {}
  );
  const [loading, setLoading] = useState(false);
  const [globalError, setGlobalError] = useState("");

  function toggleMode(mode) {
    setSelectedModes((currentModes) => {
      if (currentModes.includes(mode)) {
        return currentModes.filter((item) => item !== mode);
      }
      return [...currentModes, mode].sort(
        (left, right) => TEXT_RERANKER_MODES.indexOf(left) - TEXT_RERANKER_MODES.indexOf(right)
      );
    });
  }

  async function runComparison() {
    if (loading || !query.trim() || selectedModes.length === 0) {
      return;
    }
    setLoading(true);
    setGlobalError("");
    const loadingResults = Object.fromEntries(
      selectedModes.map((mode) => [
        mode,
        {
          loading: true,
          success: false,
          error: "",
          response: null
        }
      ])
    );
    setComparisonResults(loadingResults);

    try {
      const results = await compareTextSearchModes({
        query: query.trim(),
        topK: Number(topK),
        modes: selectedModes
      });
      const nextResults = Object.fromEntries(
        selectedModes.map((mode) => {
          const result = results[mode];
          if (result?.status === "success") {
            return [
              mode,
              {
                loading: false,
                success: true,
                error: "",
                response: result.response
              }
            ];
          }
          return [
            mode,
            {
              loading: false,
              success: false,
              error: result?.error || "请求失败。",
              response: null
            }
          ];
        })
      );
      setComparisonResults(nextResults);
      persistEvalState({
        query: query.trim(),
        topK: Number(topK),
        selectedModes,
        comparisonResponses: nextResults,
        errors: collectErrors(nextResults),
        updatedAt: new Date().toISOString()
      });
    } catch (error) {
      const message = error.message || "对比请求失败。";
      setGlobalError(message);
      persistEvalState({
        query: query.trim(),
        topK: Number(topK),
        selectedModes,
        comparisonResponses: {},
        errors: { global: message },
        updatedAt: new Date().toISOString()
      });
    } finally {
      setLoading(false);
    }
  }

  function clearComparisonState() {
    sessionStorage.removeItem(EVAL_STORAGE_KEY);
    setQuery("black shirt");
    setTopK("10");
    setSelectedModes(TEXT_RERANKER_MODES);
    setComparisonResults({});
    setGlobalError("");
  }

  return (
    <div className="page-layout evaluation-layout">
      <EvalPanel
        comparisonResults={comparisonResults}
        globalError={globalError}
        loading={loading}
        onClear={clearComparisonState}
        onQueryChange={setQuery}
        onRun={runComparison}
        onToggleMode={toggleMode}
        onTopKChange={setTopK}
        query={query}
        selectedModes={selectedModes}
        topK={topK}
      />
    </div>
  );
}

function persistEvalState(state) {
  try {
    sessionStorage.setItem(EVAL_STORAGE_KEY, JSON.stringify(state));
  } catch (error) {
    // sessionStorage may be unavailable in strict browser settings.
  }
}

function loadEvalState() {
  try {
    const rawValue = sessionStorage.getItem(EVAL_STORAGE_KEY);
    if (!rawValue) {
      return null;
    }
    return JSON.parse(rawValue);
  } catch (error) {
    sessionStorage.removeItem(EVAL_STORAGE_KEY);
    return null;
  }
}

function collectErrors(results) {
  return Object.fromEntries(
    Object.entries(results)
      .filter(([, result]) => result.error)
      .map(([mode, result]) => [mode, result.error])
  );
}

export default EvaluationPage;
