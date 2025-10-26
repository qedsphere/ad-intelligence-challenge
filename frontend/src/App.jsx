import { useState } from "react";
import axios from "axios";

export default function App() {
  const [files, setFiles] = useState([]);
  const [downloading, setDownloading] = useState(false);
  const [downloadUrl, setDownloadUrl] = useState(null);

  const handleUpload = (e) => setFiles(Array.from(e.target.files));

  const handleSubmit = async () => {
    if (!files.length) return;
    setDownloading(true);
    setDownloadUrl(null);

    const formData = new FormData();
    files.forEach((f) => formData.append("files", f));

    try {
      const res = await axios.post("http://localhost:8000/extract", formData, {
        responseType: "blob",
        onUploadProgress: (p) =>
          console.log(`Upload ${Math.round((p.loaded / p.total) * 100)}%`),
      });

      const blob = new Blob([res.data], { type: "application/zip" });
      const url = URL.createObjectURL(blob);
      setDownloadUrl(url);
    } catch (err) {
      alert("Extraction failed: " + err);
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div style={{ padding: 40, fontFamily: "sans-serif" }}>
      <h1>🧠 Ad Feature Extractor</h1>
      <input
        type="file"
        accept="image/png, image/jpeg"
        multiple
        onChange={handleUpload}
        style={{ marginBottom: 20 }}
      />
      <br />
      <button onClick={handleSubmit} disabled={downloading}>
        {downloading ? "Extracting..." : "Run Extraction"}
      </button>
      {downloadUrl && (
        <div style={{ marginTop: 20 }}>
          <a href={downloadUrl} download="extracted_features.zip">
            ⬇️ Download Extracted Features
          </a>
        </div>
      )}
    </div>
  );
}
