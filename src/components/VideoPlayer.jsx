import React, { useEffect, useRef, useState } from "react";
import videojs from "video.js";
import "video.js/dist/video-js.css";
import "videojs-youtube";
import { MdSkipPrevious, MdSkipNext, MdPlayArrow, MdPause, MdClose, MdArrowDropDown } from "react-icons/md";

// Optional: TFJS can be removed if you don't use it elsewhere
import * as tf from "@tensorflow/tfjs";

// === MediaPipe Tasks (WASM) ===
import { ObjectDetector, FilesetResolver } from "@mediapipe/tasks-vision";

export default function VideoPlayer({
  src,
  onTimeUpdate,
  onSchemaLoaded,
  onAnalysisComplete,
  modelPath = "/models/model.tflite",
  schemaLoaded = false,
}) {
  const videoElRef = useRef(null);
  const playerRef = useRef(null);

  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [current, setCurrent] = useState(0);
  const [stepMs, setStepMs] = useState(33.33);

  // Range selection
  const [selStart, setSelStart] = useState(null);
  const [selEnd, setSelEnd] = useState(null);
  const [dragging, setDragging] = useState(null);
  const sliderWrapRef = useRef(null);

  // Modal state (Annotate)
  const [isModalOpen, setModalOpen] = useState(false);
  const [tagInput, setTagInput] = useState("");
  const [modifierInput, setModifierInput] = useState("");
  const [downInput, setDownInput] = useState("");
  const [quickChosen, setQuickChosen] = useState(false);

  // Dropdown state + refs
  const [tagMenuOpen, setTagMenuOpen] = useState(false);
  const [modMenuOpen, setModMenuOpen] = useState(false);
  const [downMenuOpen, setDownMenuOpen] = useState(false);
  const tagComboRef = useRef(null);
  const modComboRef = useRef(null);
  const downComboRef = useRef(null);

  // Annotation snapshot (from timeline)
  const [annSnapshot, setAnnSnapshot] = useState([]);

  // Editable current time
  const [timeEdit, setTimeEdit] = useState("00:00.00");

  // Load Schema (CSV)
  const schemaInputRef = useRef(null);

  // Analyze state
  const [analyzing, setAnalyzing] = useState(false);
  const [analyzeMsg, setAnalyzeMsg] = useState("");

  const hasSource = !!src;

  const TAG_OPTIONS = ["SPIN", "JUKE", "WALL_MOVE", "HURDLE", "TACKLE_BIG_HIT", "STIFF_ARM", "TACKLE_DIVE", "PASS_THROW", "PASS_CATCH", "TACKLE_WRAP"];
  const MODIFIER_OPTIONS = ["LEFT", "RIGHT", "IN_AIR", "FORWARD", "ON_GROUND", "ON_SIDELINE", "ON_WALL"];

  // Receive annotation list from timeline
  useEffect(() => {
    const onState = (e) => setAnnSnapshot(Array.isArray(e.detail?.annotations) ? e.detail.annotations : []);
    window.addEventListener("at:state", onState);
    return () => window.removeEventListener("at:state", onState);
  }, []);

  const isYouTube = (url) => {
    if (!url) return false;
    try {
      const u = new URL(url);
      return ["www.youtube.com", "youtube.com", "youtu.be"].includes(u.hostname);
    } catch {
      return false;
    }
  };
  const isYT = isYouTube(src);

  const pad2 = (n) => String(n).padStart(2, "0");
  const formatTime = (s) => {
    if (!isFinite(s)) return "00:00.00";
    const totalHundredths = Math.round(s * 100);
    const min = Math.floor(totalHundredths / 6000);
    const sec = Math.floor((totalHundredths % 6000) / 100);
    const hund = totalHundredths % 100;
    return `${pad2(min)}:${pad2(sec)}.${pad2(hund)}`;
  };
  const parseTime = (str) => {
    if (str == null) return null;
    const s = String(str).trim();
    if (s === "") return null;
    if (/^\d+(\.\d+)?$/.test(s)) return Number(s);
    const m = s.match(/^(\d+):(\d{1,2})(?:\.(\d{1,3}))?$/);
    if (m) {
      const mm = Number(m[1]);
      const ss = Number(m[2]);
      let frac = m[3] ? Number(`0.${m[3]}`) : 0;
      frac = Math.round(frac * 100) / 100;
      return mm * 60 + ss + frac;
    }
    return null;
  };
  useEffect(() => {
    setTimeEdit(formatTime(current));
  }, [current]);

  // Init / update player
  useEffect(() => {
    if (!videoElRef.current || !hasSource) return;
    const options = { controls: false, preload: "auto", fluid: true, techOrder: ["html5", "youtube"], sources: [], youtube: { rel: 0, iv_load_policy: 3 } };
    const player = playerRef.current ?? (playerRef.current = videojs(videoElRef.current, options));
    player.off();

    player.on("loadedmetadata", () => setDuration(player.duration() || 0));
    player.on("timeupdate", () => {
      const t = player.currentTime() || 0;
      setCurrent(t);
      onTimeUpdate?.(t);
    });
    player.on("play", () => setIsPlaying(true));
    player.on("pause", () => setIsPlaying(false));

    player.src(isYouTube(src) ? { src, type: "video/youtube" } : { src, type: "video/mp4" });
  }, [src, hasSource, onTimeUpdate]);

  // Dispose on unmount
  useEffect(
    () => () => {
      if (playerRef.current) {
        try {
          playerRef.current.dispose();
        } catch {}
        playerRef.current = null;
      }
    },
    []
  );

  const play = async () => {
    try {
      await playerRef.current?.play();
    } catch {}
  };
  const pause = () => playerRef.current?.pause();
  const togglePlay = () => (isPlaying ? pause() : play());

  const onScrub = (e) => {
    const t = Number(e.target.value);
    const p = playerRef.current;
    if (p && !Number.isNaN(t)) {
      p.currentTime(t);
      setCurrent(t);
      onTimeUpdate?.(t);
    }
  };

  const step = (dir) => {
    const p = playerRef.current;
    if (!p) return;
    const delta = (Number(stepMs) || 0) / 1000;
    const next = Math.max(0, Math.min(duration, p.currentTime() + dir * delta));
    p.currentTime(next);
    setCurrent(next);
    onTimeUpdate?.(next);
  };

  // Hotkeys
  useEffect(() => {
    const typing = (el) => {
      if (!el) return false;
      const tag = el.tagName?.toLowerCase();
      return tag === "input" || tag === "textarea" || tag === "select" || el.isContentEditable;
    };
    const onKey = (e) => {
      if (typing(document.activeElement)) return;
      if (e.code === "Space") {
        e.preventDefault();
        togglePlay();
      }
      if (e.key === ",") {
        e.preventDefault();
        step(-1);
      }
      if (e.key === ".") {
        e.preventDefault();
        step(1);
      }
      if (e.key === "[") {
        e.preventDefault();
        if (hasSource) setSelStart(clamp(current, 0, duration));
      }
      if (e.key === "]") {
        e.preventDefault();
        if (hasSource) {
          const t = clamp(current, 0, duration);
          setSelEnd(t);
          playerRef.current?.pause();
          setIsPlaying(false);
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [isPlaying, stepMs, duration, current, hasSource]);

  // Selection helpers
  const clamp = (v, min, max) => Math.max(min, Math.min(max, v));
  const clientXToTime = (clientX) => {
    const el = sliderWrapRef.current;
    if (!el || duration <= 0) return 0;
    const rect = el.getBoundingClientRect();
    const x = clamp(clientX - rect.left, 0, rect.width);
    const pct = rect.width === 0 ? 0 : x / rect.width;
    return clamp(pct * duration, 0, duration);
  };
  const clearSelection = () => {
    setSelStart(null);
    setSelEnd(null);
    setDragging(null);
  };
  const onOverlayMouseDown = (e) => {
    if (!hasSource) return;
    if (e.shiftKey) {
      const t = clientXToTime(e.clientX);
      setSelStart(t);
      setSelEnd(t);
      setDragging("end");
      e.preventDefault();
      return;
    }
    const t = clientXToTime(e.clientX);
    const p = playerRef.current;
    if (p) {
      p.currentTime(t);
      setCurrent(t);
      onTimeUpdate?.(t);
    }
  };
  const onStartHandleMouseDown = (e) => {
    if (!hasSource) return;
    setDragging("start");
    e.preventDefault();
    e.stopPropagation();
  };
  const onEndHandleMouseDown = (e) => {
    if (!hasSource) return;
    setDragging("end");
    e.preventDefault();
    e.stopPropagation();
  };
  useEffect(() => {
    const onMove = (e) => {
      if (!dragging) return;
      const t = clientXToTime(e.clientX);
      if (dragging === "start") {
        setSelStart(() => {
          const end = selEnd ?? t;
          if (t > end) {
            setSelEnd(t);
            return end;
          }
          return t;
        });
      } else if (dragging === "end") {
        setSelEnd(() => {
          const start = selStart ?? t;
          if (t < start) {
            setSelStart(t);
            return start;
          }
          return t;
        });
      }
    };
    const onUp = () => setDragging(null);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [dragging, selStart, selEnd, duration]);

  // Close dropdowns when clicking outside or pressing Escape
  useEffect(() => {
    const onDocClick = (e) => {
      if (!isModalOpen) return;
      if (tagComboRef.current && !tagComboRef.current.contains(e.target)) setTagMenuOpen(false);
      if (modComboRef.current && !modComboRef.current.contains(e.target)) setModMenuOpen(false);
      if (downComboRef.current && !downComboRef.current.contains(e.target)) setDownMenuOpen(false);
    };
    const onEsc = (e) => {
      if (e.key === "Escape") {
        setTagMenuOpen(false);
        setModMenuOpen(false);
        setDownMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", onDocClick);
    document.addEventListener("keydown", onEsc);
    return () => {
      document.removeEventListener("mousedown", onDocClick);
      document.removeEventListener("keydown", onEsc);
    };
  }, [isModalOpen]);

  const pct = (t) => (duration > 0 ? (t / duration) * 100 : 0);
  const hasSelection = selStart != null && selEnd != null && duration > 0;
  const hasStart = selStart != null && duration > 0;
  const hasEnd = selEnd != null && duration > 0;
  const rangeL = hasStart ? (hasEnd ? Math.min(selStart, selEnd) : selStart) : 0;
  const rangeR = hasSelection ? Math.max(selStart, selEnd) : 0;

  // CSV schema parsing
  const parseSchemaTime = (raw) => {
    if (raw == null) return null;
    const s = String(raw).trim();
    if (s === "") return null;

    if (s.includes(".")) {
      const [mStr, sStr] = s.split(".");
      const minutes = Number(mStr);
      const seconds = Number(sStr);
      if (!Number.isFinite(minutes) || !Number.isFinite(seconds)) return null;
      if (seconds < 0 || seconds > 59) return null;
      return minutes * 60 + seconds;
    }

    if (/^\d{2,3}$/.test(s)) {
      const n = Number(s);
      if (s.length === 2) {
        const seconds = n % 100;
        return seconds;
      }
      const minutes = Math.floor(n / 100);
      const seconds = n % 100;
      if (seconds > 59) return null;
      return minutes * 60 + seconds;
    }

    if (/^\d+$/.test(s)) {
      const minutes = Number(s);
      return minutes * 60;
    }

    return null;
  };

  // Load schema file
  const handleSchemaFile = async (file) => {
    if (!file) return;
    const text = await file.text();

    // Clear current rows before importing a new CSV
    window.dispatchEvent(new CustomEvent("vp:clear-annotations"));

    const lines = text.replace(/\r\n?/g, "\n").split("\n").filter(Boolean);

    const parseCSVLine = (line) => {
      const out = [];
      let cur = "",
        inQ = false;
      for (let i = 0; i < line.length; i++) {
        const ch = line[i];
        if (ch === '"') {
          if (inQ && line[i + 1] === '"') {
            cur += '"';
            i++;
          } else {
            inQ = !inQ;
          }
        } else if (ch === "," && !inQ) {
          out.push(cur);
          cur = "";
        } else {
          cur += ch;
        }
      }
      out.push(cur);
      return out.map((s) => s.trim());
    };

    const header = parseCSVLine(lines[0]).map((h) => h.toLowerCase());
    const idxTag = header.indexOf("tagname");
    const idxStart = header.indexOf("starttime");
    const idxEnd = header.indexOf("endtime");
    const idxMod = header.indexOf("modifiers");

    if (idxTag === -1 || idxStart === -1 || idxEnd === -1 || idxMod === -1) {
      console.warn("CSV missing required headers: TagName, StartTime, EndTime, Modifiers");
      return;
    }

    for (let i = 1; i < lines.length; i++) {
      const cols = parseCSVLine(lines[i]);
      if (!cols.length) continue;
      const tag = (cols[idxTag] ?? "").trim();
      const startStr = cols[idxStart];
      const endStr = cols[idxEnd];
      const mod = (cols[idxMod] ?? "").trim();

      const startSec = parseSchemaTime(startStr);
      const endSec = parseSchemaTime(endStr);

      if (!tag || startSec == null || !isFinite(startSec)) continue;

      const detail = {
        start: startSec,
        end: endSec != null && isFinite(endSec) ? endSec : startSec,
        tag,
        modifier: mod,
      };

      window.dispatchEvent(new CustomEvent("vp:add-annotation", { detail }));
    }

    // Notify parent for header pill
    onSchemaLoaded?.(file.name);
  };

  const onClickLoadSchema = () => schemaInputRef.current?.click();
  const onSchemaChange = (e) => {
    const f = e.target.files?.[0];
    if (f) handleSchemaFile(f);
    e.target.value = "";
  };

  // Modal helpers
  const openAnnotate = () => {
    setTagInput("");
    setModifierInput("");
    setDownInput("");
    setQuickChosen(false);
    setTagMenuOpen(false);
    setModMenuOpen(false);
    setDownMenuOpen(false);
    setModalOpen(true);
  };
  const addAnnotationFromModal = () => {
    if (!hasStart || !hasEnd || !tagInput.trim() || conflict) return;
    const detail = { start: rangeL, end: rangeR, tag: tagInput.trim(), modifier: (modifierInput || "").trim(), down: (downInput || "").trim() };
    window.dispatchEvent(new CustomEvent("vp:add-annotation", { detail }));

    // advance and reset
    const p = playerRef.current;
    const delta = Math.max((Number(stepMs) || 0) / 1000, 1 / FPS);
    const nextT = clamp(rangeR + delta + 1e-4, 0, duration);
    if (p) p.currentTime(nextT);
    setCurrent(nextT);
    onTimeUpdate?.(nextT);
    clearSelection();
    setModalOpen(false);
  };

  // Conflict detection aligned to timeline quantization
  const FPS = 30;
  const frameKey = (t) => Math.round(t * FPS) / FPS;
  const EPS = 1 / FPS / 2;

  const startKey = frameKey(rangeL);
  const conflict = annSnapshot.some((a) => {
    const s = a.startKey;
    const e = a.endKey ?? a.startKey;
    const lo = Math.min(s, e);
    const hi = Math.max(s, e);
    if (startKey > lo && startKey < hi) return true;
    const nearLo = Math.abs(startKey - lo) <= EPS;
    const nearHi = Math.abs(startKey - hi) <= EPS;
    return nearLo || nearHi;
  });

  const commitTimeEdit = () => {
    const p = playerRef.current;
    if (!p) return;
    const parsed = parseTime(timeEdit);
    if (parsed == null || !isFinite(parsed)) {
      setTimeEdit(formatTime(current));
      return;
    }
    const clamped = Math.max(0, Math.min(duration || 0, parsed));
    const rounded = Math.round(clamped * 100) / 100;
    p.currentTime(rounded);
    setCurrent(rounded);
    onTimeUpdate?.(rounded);
    setTimeEdit(formatTime(rounded));
  };

  // =================== ANALYSIS WITH MEDIAPIPE TASKS ===================

  const detectorRef = useRef(null);

  const ensureDetector = async () => {
    if (detectorRef.current) return detectorRef.current;

    setAnalyzeMsg("Loading detector…");

    // Resolve WASM assets (official CDN)
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");

    const detector = await ObjectDetector.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: modelPath,
        delegate: "CPU",
      },
      scoreThreshold: 0.5,
      maxResults: 25,
      runningMode: "IMAGE",
    });

    detectorRef.current = detector;
    return detector;
  };

  // Grab a frame at exact second and return a canvas (Tasks accepts HTMLCanvasElement)
  const grabCanvasAt = (sec) =>
    new Promise((resolve, reject) => {
      const p = playerRef.current;
      const videoEl = videoElRef.current;
      if (!p || !videoEl) return reject(new Error("Player/video not ready"));
      const onSeeked = () => {
        try {
          const vw = videoEl.videoWidth,
            vh = videoEl.videoHeight;
          if (!vw || !vh) throw new Error("No video dimension");
          const canvas = document.createElement("canvas");
          canvas.width = vw;
          canvas.height = vh;
          const ctx = canvas.getContext("2d");
          ctx.drawImage(videoEl, 0, 0, vw, vh);
          resolve(canvas);
        } catch (err) {
          reject(err);
        } finally {
          videoEl.removeEventListener("seeked", onSeeked);
        }
      };
      videoEl.addEventListener("seeked", onSeeked, { once: true });
      p.currentTime(sec);
    });

  // Label mapping helpers
  const FALLBACK_BY_INDEX = {
    0: "BALL_CARRIER",
    1: "DEFENDER",
    2: "ATTACKER",
  };
  const BALL_CARRIER_LABELS = new Set(["BALL_CARRIER", "ball_carrier", "carrier"]);
  const DEFENDER_LABELS = new Set(["DEFENDER", "defender"]);
  const ATTACKER_LABELS = new Set(["ATTACKER", "attacker"]);

  const labelOf = (det) => {
    const c = det.categories?.[0];
    const name = c?.categoryName;
    if (name && name.trim()) return name.trim();
    const idx = Number.isFinite(c?.index) ? c.index : undefined;
    if (idx != null && FALLBACK_BY_INDEX[idx]) return FALLBACK_BY_INDEX[idx];
    return "";
  };

  const countDefendersInFront = (detections) => {
    const candidates = detections.map((det) => {
      const label = labelOf(det);
      const score = det.categories?.[0]?.score ?? 0;
      const bb = det.boundingBox;
      return { label, score, bb };
    });

    const carriers = candidates.filter((c) => BALL_CARRIER_LABELS.has(c.label));
    const defenders = candidates.filter((c) => DEFENDER_LABELS.has(c.label));

    if (carriers.length === 0) return 0;
    let carrier = carriers[0];
    for (const c of carriers) if (c.score > carrier.score) carrier = c;

    const cy = carrier.bb.originY + carrier.bb.height / 2;
    let count = 0;
    for (const d of defenders) {
      const dy = d.bb.originY + d.bb.height / 2;
      if (dy < cy) count++;
    }
    return count;
  };

  const runAnalysis = async () => {
    if (isYT || !schemaLoaded || analyzing) return;
    try {
      setAnalyzing(true);
      setAnalyzeMsg("Preparing…");

      await tf.ready();
      const detector = await ensureDetector();

      const rows = annSnapshot.filter((a) => Number.isFinite(a.startKey));
      if (rows.length === 0) {
        setAnalyzeMsg("No annotated windows found.");
        setAnalyzing(false);
        return;
      }

      const results = {}; // { TAG: [counts...] }

      for (let i = 0; i < rows.length; i++) {
        const row = rows[i];
        setAnalyzeMsg(`Analyzing ${i + 1}/${rows.length}…`);

        const canvas = await grabCanvasAt(row.startKey);
        const dets = await detector.detect(canvas);
        const count = countDefendersInFront(dets.detections || []);

        if (!results[row.tagName]) results[row.tagName] = [];
        results[row.tagName].push(count);
      }

      setAnalyzeMsg("Done.");
      onAnalysisComplete?.(results);
    } catch (err) {
      console.error(err);
      setAnalyzeMsg("Analysis failed (see console).");
    } finally {
      setAnalyzing(false);
      setTimeout(() => setAnalyzeMsg(""), 1500);
    }
  };

  // =================== RENDER ===================

  if (!hasSource) return null;

  return (
    <section className="vp">
      {/* Player */}
      <div className="vp__videowrap">
        <video ref={videoElRef} className="video-js vjs-default-skin vp__videojs" playsInline />
      </div>

      {/* Toolbar */}
      <div className="vp__controls">
        <div className="vp__toolbar" style={{ display: "grid", gridTemplateColumns: "1fr auto 1fr", alignItems: "center", gap: "12px" }}>
          <div className="vp__time" style={{ justifySelf: "start", display: "flex", alignItems: "center", gap: 6 }}>
            <input
              className="mono"
              value={timeEdit}
              onChange={(e) => setTimeEdit(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  commitTimeEdit();
                }
                if (e.key === "Escape") {
                  setTimeEdit(formatTime(current));
                }
              }}
              onBlur={() => setTimeEdit(formatTime(current))}
              aria-label="Current time (editable)"
              style={{ width: 84, padding: "2px 6px", borderRadius: 6, border: "1px solid var(--line)", background: "#0f1420", color: "var(--text)" }}
              inputMode="numeric"
              placeholder="mm:ss.hh"
            />
            <span className="vp__timeSep">/</span>
            <span className="vp__timeTotal mono">{formatTime(duration)}</span>
          </div>

          <div className="vp__btnGroup" role="group" aria-label="Playback" style={{ justifySelf: "center" }}>
            <button title="Frame Back (,)" onClick={() => step(-1)} aria-label="Frame Back">
              <MdSkipPrevious size={22} />
            </button>
            <button title={isPlaying ? "Pause (Space)" : "Play (Space)"} onClick={togglePlay} aria-label={isPlaying ? "Pause" : "Play"}>
              {isPlaying ? <MdPause size={22} /> : <MdPlayArrow size={22} />}
            </button>
            <button title="Frame Forward (.)" onClick={() => step(1)} aria-label="Frame Forward">
              <MdSkipNext size={22} />
            </button>
          </div>

          <div style={{ justifySelf: "end", display: "flex", gap: 8, alignItems: "center" }}>
            {analyzeMsg && (
              <span className="mono" style={{ color: "var(--muted)", fontSize: 12 }}>
                {analyzeMsg}
              </span>
            )}

            <input ref={schemaInputRef} type="file" accept=".csv,text/csv" hidden onChange={onSchemaChange} />
            <button className="src__btn" type="button" onClick={onClickLoadSchema} title="Load annotation schema CSV">
              Load Schema
            </button>

            <button
              className="src__btn"
              type="button"
              onClick={runAnalysis}
              disabled={isYT || !schemaLoaded || analyzing}
              title={isYT ? "YouTube sources are preview-only; analysis disabled" : !schemaLoaded ? "Load a schema to enable Analyze" : analyzing ? "Analyzing…" : "Analyze"}
            >
              {analyzing ? "Analyzing…" : "Analyze"}
            </button>
          </div>
        </div>

        {/* Scrubber + overlay */}
        <div className="vp__slider">
          <div className="vp__scrubwrap" ref={sliderWrapRef}>
            <input type="range" min={0} max={duration || 0} step={0.001} value={current} onChange={onScrub} className="vp__scrub" />
            <div className="vp__rangeOverlay" onMouseDown={onOverlayMouseDown} title="Shift+Drag to select a time range. Click to move playhead.">
              {hasSelection && (
                <div className="vp__rangeBar" style={{ left: `${pct(Math.min(selStart, selEnd))}%`, width: `${Math.max(0, pct(Math.max(selStart, selEnd)) - pct(Math.min(selStart, selEnd)))}%` }} />
              )}
              {selStart != null && (
                <div className={`vp__handle vp__handle--start ${dragging === "start" ? "is-dragging" : ""}`} style={{ left: `${pct(selStart)}%` }} onMouseDown={onStartHandleMouseDown} />
              )}
              {selEnd != null && (
                <div className={`vp__handle vp__handle--end ${dragging === "end" ? "is-dragging" : ""}`} style={{ left: `${pct(selEnd)}%` }} onMouseDown={onEndHandleMouseDown} />
              )}
            </div>
          </div>
        </div>

        {/* Range row — left-aligned chips + Annotate */}
        <div className="vp__rangeRow">
          {hasStart ? (
            <div className="vp__chips">
              <button
                type="button"
                className="chip chip--clickable"
                onClick={() => {
                  playerRef.current?.currentTime(rangeL);
                  setCurrent(rangeL);
                }}
                title="Jump to Start"
              >
                Start {formatTime(rangeL)}
              </button>
              <button
                type="button"
                className={`chip ${hasEnd ? "chip--clickable" : "chip--disabled"}`}
                onClick={() => {
                  if (!hasEnd) return;
                  playerRef.current?.currentTime(rangeR);
                  setCurrent(rangeR);
                }}
                title={hasEnd ? "Jump to End" : "Set End (])"}
                disabled={!hasEnd}
              >
                {hasEnd ? `End ${formatTime(rangeR)}` : "End --:--.--"}
              </button>
              <button className={`chip ${hasEnd ? "chip--clickable" : "chip--disabled"}`} type="button" disabled={!hasEnd} onClick={openAnnotate} title={hasEnd ? "Annotate range" : "Set End (])"}>
                Annotate
              </button>
              <button className="chipBtn chipBtn--danger" title="Clear range" onClick={clearSelection}>
                <MdClose size={18} />
              </button>
            </div>
          ) : (
            <div className="vp__hint mono">
              Tip: <kbd>Shift</kbd> + drag to select a range. Use <kbd>[</kbd> and <kbd>]</kbd> to set Start/End.
            </div>
          )}

          <details className="vp__adv">
            <summary>Advanced</summary>
            <label className="vp__stepLabel">
              Step (ms): <input type="number" min="1" step="1" value={stepMs} onChange={(e) => setStepMs(e.target.value)} className="step-input" />
            </label>
            <span className="hint">33.33 ≈ 30fps · 16.67 ≈ 60fps</span>
          </details>
        </div>
      </div>

      {/* Annotate Modal */}
      {isModalOpen && (
        <div className="modalOverlay" onMouseDown={(e) => { if (e.target.classList.contains("modalOverlay")) setModalOpen(false); }} style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.5)", display: "grid", placeItems: "center", zIndex: 1000 }}>
          <div className="modalCard modalCard--fit" role="dialog" aria-modal="true" aria-labelledby="annTitle" style={{ background:"var(--panel)", border:"1px solid var(--line)", borderRadius:12, padding:16, position:"relative" }}>
            <button onClick={() => setModalOpen(false)} title="Close" aria-label="Close" style={{ position: "absolute", top: 8, right: 8, background: "#4a2b2b", color: "#ffb3b3", border: "1px solid #5a3434", borderRadius: 8, width: 32, height: 32, display: "grid", placeItems: "center", cursor: "pointer" }}>
              <MdClose size={18} />
            </button>
            <h3 id="annTitle" style={{ margin: "0 0 12px 0", fontSize: 16 }}>Create Annotation</h3>

            {/* Quick tags */}
            <div className="annQuick">
              <button className="src__btn" onClick={() => { setTagInput("SPIN"); setQuickChosen(true); }} disabled={quickChosen || conflict}>+ Spin (S)</button>
              <button className="src__btn" onClick={() => { setTagInput("JUKE"); setQuickChosen(true); }} disabled={quickChosen || conflict}>+ Juke (J)</button>
              <button className="src__btn" onClick={() => { setTagInput("TACKLE"); setQuickChosen(true); }} disabled={quickChosen || conflict}>+ Tackle (T)</button>
              <button className="src__btn" onClick={() => { setTagInput(""); setQuickChosen(true); }} disabled={quickChosen || conflict}>+ Custom Action</button>
            </div>

            {/* Times inline */}
            <div style={{ display: "flex", gap: 16, alignItems: "center", marginBottom: 12, color: "var(--muted)", fontSize: 13 }}>
              <div>
                Start: <span className="mono" style={{ color: "var(--text)" }}>{formatTime(rangeL)}</span>
              </div>
              <div>
                End: <span className="mono" style={{ color: "var(--text)" }}>{formatTime(rangeR)}</span>
              </div>
            </div>

            {/* Fields (stacked) */}
            <div className="annFieldsStack">
              <label className="annField">
                <span className="annField__label">Tag name</span>
                <div className="combo" ref={tagComboRef}>
                  <input type="text" className="src__input combo__input" value={tagInput} onChange={(e) => setTagInput(e.target.value)} placeholder="SPIN / JUKE / TACKLE / custom…" disabled={conflict} aria-label="Tag name" />
                  <button type="button" className="combo__btn" onClick={() => setTagMenuOpen((v) => !v)} aria-haspopup="listbox" aria-expanded={tagMenuOpen} disabled={conflict}>
                    <MdArrowDropDown size={18} />
                  </button>
                  {tagMenuOpen && (
                    <div className="combo__menu" role="listbox">
                      {TAG_OPTIONS.map((opt) => (
                        <button key={opt} type="button" className="combo__option" onClick={() => { setTagInput(opt); setTagMenuOpen(false); }}>
                          {opt}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </label>

              <label className="annField">
                <span className="annField__label">Modifier</span>
                <div className="combo" ref={modComboRef}>
                  <input type="text" className="src__input combo__input" value={modifierInput} onChange={(e) => setModifierInput(e.target.value)} placeholder="Optional details" disabled={conflict} aria-label="Modifier" />
                  <button type="button" className="combo__btn" onClick={() => setModMenuOpen((v) => !v)} aria-haspopup="listbox" aria-expanded={modMenuOpen} disabled={conflict}>
                    <MdArrowDropDown size={18} />
                  </button>
                  {modMenuOpen && (
                    <div className="combo__menu" role="listbox">
                      {MODIFIER_OPTIONS.map((opt) => (
                        <button key={opt} type="button" className="combo__option" onClick={() => { setModifierInput(opt); setModMenuOpen(false); }}>
                          {opt}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </label>

              <label className="annField">
                <span className="annField__label">Down</span>
                <div className="combo" ref={downComboRef}>
                  <input type="text" className="src__input combo__input" value={downInput} onChange={(e) => setDownInput(e.target.value)} placeholder="1 / 2 / 3 / 4" disabled={conflict} aria-label="Down" inputMode="numeric" />
                  <button type="button" className="combo__btn" onClick={() => setDownMenuOpen((v) => !v)} aria-haspopup="listbox" aria-expanded={downMenuOpen} disabled={conflict}>
                    <MdArrowDropDown size={18} />
                  </button>
                  {downMenuOpen && (
                    <div className="combo__menu" role="listbox">
                      {[1, 2, 3, 4].map((opt) => (
                        <button key={opt} type="button" className="combo__option" onClick={() => { setDownInput(String(opt)); setDownMenuOpen(false); }}>
                          {opt}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </label>
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: 8, justifyContent: "space-between" }}>
              {conflict && <div style={{ color: "#ffb3b3" }}>Annotation already exists for given start time.</div>}
              <div style={{ marginLeft: "auto", display: "flex", gap: 8 }}>
                <button className="src__btn" onClick={() => setModalOpen(false)}>Cancel</button>
                <button className="src__btn" onClick={addAnnotationFromModal} disabled={!hasEnd || !hasStart || !tagInput.trim() || conflict} title={!tagInput.trim() ? "Enter a tag name" : "Add annotation"}>+ Add</button>
              </div>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
