import React, { useState } from "react";

export default function Loader({ onLoad }) {
  const [tab, setTab] = useState("file");
  const [urlInput, setUrlInput] = useState("");

  const isYouTube = (url) => {
    try { const u = new URL(url); return ["www.youtube.com","youtube.com","youtu.be"].includes(u.hostname); } catch { return false; }
  };

  const onFileChange = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const src = URL.createObjectURL(file);
    onLoad?.({ src, name: file.name, isURL: false });
  };

  const onDropFile = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (!file) return;
    const src = URL.createObjectURL(file);
    onLoad?.({ src, name: file.name, isURL: false });
  };

  const loadFromUrl = () => {
    const trimmed = urlInput.trim();
    if (!trimmed) return;
    onLoad?.({ src: trimmed, name: trimmed, isURL: true });
  };

  return (
    <section className="src">
      <div className="src__tabs" role="tablist" aria-label="Video source">
        <button className={`src__tab ${tab==="file"?"is-active":""}`} role="tab" aria-selected={tab==="file"} onClick={()=>setTab("file")}>Local file</button>
        <button className={`src__tab ${tab==="url"?"is-active":""}`} role="tab" aria-selected={tab==="url"} onClick={()=>setTab("url")}>YouTube URL</button>
      </div>

      {tab==="file" && (
        <div className="src__panel" role="tabpanel" aria-label="Local file">
          <label className="sr-only" htmlFor="file-input">Choose a video file</label>
          <div className="src__drop" onDragOver={(e)=>{e.preventDefault(); e.dataTransfer.dropEffect="copy";}} onDrop={onDropFile}>
            <div className="src__dropInner">
              <div className="src__dropIcon">📦</div>
              <div className="src__dropText"><strong>Drag & drop a video</strong><span>or</span></div>
              <label className="src__btn">Choose file<input id="file-input" type="file" accept="video/*" onChange={onFileChange} hidden /></label>
            </div>
          </div>
          <div className="src__hint">Supported: mp4, mov, webm.</div>
        </div>
      )}

      {tab==="url" && (
        <div className="src__panel" role="tabpanel" aria-label="YouTube URL">
          <label className="src__label" htmlFor="url-input">YouTube URL</label>
          <div className="src__urlrow">
            <input
              id="url-input"
              type="text"
              placeholder="https://www.youtube.com/watch?v=..."
              value={urlInput}
              onChange={(e)=>setUrlInput(e.target.value)}
              onKeyDown={(e)=>{ if (e.key==="Enter") loadFromUrl(); }}
              className="src__input"
            />
            <button className="src__btn" onClick={loadFromUrl} disabled={!urlInput.trim()}>Load</button>
          </div>
          <div className="src__hint">YouTube videos can be previewed but not analyzed due to browser restrictions.</div>
        </div>
      )}
    </section>
  );
}
