import { useEffect, useState } from "react";
import { IMAGE_RERANKER_MODES } from "../api/searchApi.js";

const modeLabels = {
  none: "Vector only",
  rule: "Rule reranker default"
};

function ImageUploadBox({
  onSearch,
  loading,
  topK,
  rerankerType,
  imageFileName,
  clearToken,
  onTopKChange,
  onRerankerTypeChange,
  onImageFileNameChange
}) {
  const [file, setFile] = useState(null);
  const currentMode = IMAGE_RERANKER_MODES.includes(rerankerType) ? rerankerType : "rule";
  const topKNumber = Number(topK);
  const topKIsValid = topKNumber >= 1 && topKNumber <= 100;

  useEffect(() => {
    setFile(null);
  }, [clearToken]);

  function handleSubmit(event) {
    event.preventDefault();
    if (!file) {
      return;
    }
    if (!topKIsValid) {
      return;
    }
    onSearch({
      file,
      topK: topKNumber,
      rerankerType: currentMode,
      imageFileName: file.name
    });
  }

  function handleFileChange(event) {
    const nextFile = event.target.files?.[0] || null;
    setFile(nextFile);
    onImageFileNameChange(nextFile?.name || "");
  }

  return (
    <form className="search-box" onSubmit={handleSubmit}>
      <label>
        上传图片
        <input
          accept="image/png,image/jpeg,image/webp"
          type="file"
          onChange={handleFileChange}
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
          {IMAGE_RERANKER_MODES.map((mode) => (
            <label
              className={currentMode === mode ? "mode-option selected" : "mode-option"}
              key={mode}
            >
              <input
                checked={currentMode === mode}
                name="image-reranker-type"
                onChange={() => onRerankerTypeChange(mode)}
                type="radio"
              />
              <span>{modeLabels[mode]}</span>
            </label>
          ))}
        </div>
      </fieldset>

      <p className="form-hint">支持 jpg / png / webp，请上传真实图片文件。</p>
      <p className="form-hint">trained / pairwise currently support text search only.</p>
      {imageFileName && !file ? (
        <p className="form-hint">
          上次图片：{imageFileName}。图片文件不会在页面切换后保留，如需再次搜索请重新上传。
        </p>
      ) : null}

      <button type="submit" disabled={loading || !file || !topKIsValid}>
        图片搜索
      </button>
    </form>
  );
}

export default ImageUploadBox;
