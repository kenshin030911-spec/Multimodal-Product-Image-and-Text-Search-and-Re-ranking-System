import { useState } from "react";
import SearchPage from "./pages/SearchPage.jsx";
import EvaluationPage from "./pages/EvaluationPage.jsx";

function App() {
  const [activePage, setActivePage] = useState("search");

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <h1>多模态商品图文搜索与重排系统</h1>
        </div>
        <nav className="tabs" aria-label="页面切换">
          <button
            className={activePage === "search" ? "tab active" : "tab"}
            type="button"
            onClick={() => setActivePage("search")}
          >
            搜索
          </button>
          <button
            className={activePage === "evaluation" ? "tab active" : "tab"}
            type="button"
            onClick={() => setActivePage("evaluation")}
          >
            评估
          </button>
        </nav>
      </header>

      <main>
        {activePage === "search" ? <SearchPage /> : <EvaluationPage />}
      </main>
    </div>
  );
}

export default App;
