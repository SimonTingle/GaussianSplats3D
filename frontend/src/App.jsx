import React, { useState } from 'react';
import axios from 'axios';
import { Upload, FileDown, Box, Loader2, AlertCircle } from 'lucide-react';
import './App.css';

// --- CONFIGURATION ---
// If running locally, use localhost. If on cloud, use relative path or cloud URL.
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8080";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [downloadUrl, setDownloadUrl] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
      setDownloadUrl(null);
      setError(null);
    }
  };

  const handleGenerate = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setDownloadUrl(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_URL}/generate`, formData, {
        responseType: 'blob', // Important for file downloads
      });

      // Create a link to download the blob
      const url = window.URL.createObjectURL(new Blob([response.data]));
      setDownloadUrl(url);
    } catch (err) {
      console.error(err);
      setError("Failed to generate model. Ensure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <header className="header">
        <Box size={32} className="logo-icon" />
        <h1>GaussianSplats3D</h1>
      </header>

      <main className="main-content">
        <div className="card">
          {/* Upload Section */}
          <div className={`upload-zone ${!preview ? 'empty' : ''}`}>
            {preview ? (
              <img src={preview} alt="Preview" className="image-preview" />
            ) : (
              <div className="upload-placeholder">
                <Upload size={48} />
                <p>Drag & drop or click to upload an image</p>
              </div>
            )}
            <input type="file" onChange={handleFileChange} accept="image/*" className="file-input" />
          </div>

          {/* Controls */}
          <div className="controls">
            {error && (
              <div className="error-banner">
                <AlertCircle size={20} />
                <span>{error}</span>
              </div>
            )}

            {!downloadUrl ? (
              <button 
                onClick={handleGenerate} 
                disabled={!file || loading}
                className="btn-primary"
              >
                {loading ? (
                  <>
                    <Loader2 className="spin" size={20} /> Generating 3D Model...
                  </>
                ) : (
                  "Generate 3D Splat"
                )}
              </button>
            ) : (
              <a 
                href={downloadUrl} 
                download="model.ply" 
                className="btn-success"
              >
                <FileDown size={20} /> Download .PLY Model
              </a>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
