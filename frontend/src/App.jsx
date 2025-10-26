import { useState } from "react";
import axios from "axios";
import "./App.css";

export default function App() {
  const [files, setFiles] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [downloadUrl, setDownloadUrl] = useState(null);
  const [summaries, setSummaries] = useState([]);

  const handleFileChange = (e) => {
    const selectedFiles = Array.from(e.target.files);
    setFiles((prevFiles) => [...prevFiles, ...selectedFiles]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!files.length) return alert("Please upload at least one file.");

    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));

    setIsLoading(true);
    setDownloadUrl(null);
    setSummaries([]);

    try {
      const response = await axios.post("http://localhost:8000/extract", formData, {
        responseType: "blob",
      });

      const blob = new Blob([response.data]);
      const url = URL.createObjectURL(blob);
      setDownloadUrl(url);

      // fetch summaries after extraction
      const summariesRes = await axios.get("http://localhost:8000/summaries");
      setSummaries(summariesRes.data.summaries);
    } catch (err) {
      console.error(err);
      alert("Error during extraction.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <h1>Ad Intelligence Feature Extractor</h1>

      <form className="upload-box" onSubmit={handleSubmit}>
        <label className="file-input">
          <input type="file" multiple onChange={handleFileChange} />
          <p>Drag and drop or click to select image files</p>
        </label>

        {files.length > 0 && (
          <div className="file-list">
            <h3>Selected Files</h3>
            <ul>
              {files.map((file, idx) => (
                <li key={idx}>{file.name}</li>
              ))}
            </ul>
          </div>
        )}

        <button type="submit" disabled={isLoading}>
          {isLoading ? "Extracting..." : "Run Extraction"}
        </button>
      </form>

      {downloadUrl && (
        <a className="download-link" href={downloadUrl} download="extracted_features.zip">
          Download Extracted Features
        </a>
      )}

      {summaries.length > 0 && (
        <div className="summary-container">
          <h2>Extraction Summaries</h2>
          {summaries.map((item, idx) => (
            <div key={idx} className="summary-box">
              <h3>{item.file}</h3>
              <pre>{item.summary}</pre>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
