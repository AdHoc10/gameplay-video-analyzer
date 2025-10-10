import React, { useEffect, useMemo, useState } from "react";
import { MdChevronLeft, MdChevronRight } from "react-icons/md";

export default function AnnotationTimeline({ currentTime = 0, hasSource }) {
  const [annotations, setAnnotations] = useState([]);

  useEffect(() => { if (!hasSource) setAnnotations([]); }, [hasSource]);

  useEffect(() => {
    const onClear = () => setAnnotations([]);
    window.addEventListener("vp:clear-annotations", onClear);
    return () => window.removeEventListener("vp:clear-annotations", onClear);
  }, []);

  if (!hasSource) return null;

  const FPS = 30;
  const frameKey = (t) => (Number.isFinite(t) ? Math.round(t * FPS) / FPS : 0);
  const currentFrameKey = useMemo(() => frameKey(currentTime), [currentTime]);

  const normalize = (s) => (s ?? "").trim().toLowerCase();

  const fmtClock = (s) => {
    if (!isFinite(s)) return "00:00.00";
    const ms = Math.floor((s % 1) * 100), sec = Math.floor(s) % 60, min = Math.floor(s / 60);
    return `${String(min).padStart(2,"0")}:${String(sec).padStart(2,"0")}.${String(ms).padStart(2,"0")}`;
  };

  const fmtMdotS = (s) => {
    if (!isFinite(s)) return "0.00";
    const total = Math.floor(s);
    const m = Math.floor(total / 60);
    const sec = total % 60;
    return `${m}.${String(sec).padStart(2, "0")}`;
  };

  const hasTagAtStart = (tagName, startKey) =>
    annotations.some((a) => a.startKey === startKey && normalize(a.tagName) === normalize(tagName));

  useEffect(() => {
    window.dispatchEvent(new CustomEvent("at:state", { detail: { annotations } }));
  }, [annotations]);

  // Accept annotations from player (now includes 'down')
  useEffect(() => {
    const handler = (e) => {
      const { start, end, tag, modifier, down } = e.detail || {};
      if (start == null || !isFinite(start)) return;
      const startKey = frameKey(start);
      const endKey = isFinite(end) ? frameKey(end) : null;
      const tagName = (tag ?? "").trim();
      const mod = (modifier ?? "").trim();
      const dn = (down ?? "").trim();
      if (tagName && hasTagAtStart(tagName, startKey)) return;

      const newAnn = {
        id: crypto?.randomUUID?.() ?? Date.now(),
        startKey,
        endKey,
        startTime: fmtClock(startKey),
        endTime: endKey != null ? fmtClock(endKey) : "",
        tagName,
        modifier: mod,
        down: dn,
      };
      setAnnotations((prev) => {
        const next = [...prev, newAnn];
        next.sort((a, b) => a.startKey - b.startKey);
        return next;
      });
    };
    window.addEventListener("vp:add-annotation", handler);
    return () => window.removeEventListener("vp:add-annotation", handler);
  }, [annotations]);

  // Keyboard quick-adds
  const ACTIONS = { s: "SPIN", j: "JUKE", t: "TACKLE" };
  useEffect(() => {
    const typing = (el) => {
      if (!el) return false;
      const tag = el.tagName?.toLowerCase();
      return tag === "input" || tag === "textarea" || tag === "select" || el.isContentEditable;
    };
    const onKey = (e) => {
      if (typing(document.activeElement)) return;
      const k = e.key.toLowerCase();
      const tagName = ACTIONS[k];
      if (!tagName) return;
      const startKey = currentFrameKey;
      if (hasTagAtStart(tagName, startKey)) return;
      e.preventDefault();
      const newAnn = {
        id: crypto?.randomUUID?.() ?? Date.now(),
        startKey,
        endKey: null,
        startTime: fmtClock(startKey),
        endTime: "",
        tagName,
        modifier: "",
        down: "",
      };
      setAnnotations((prev) => {
        const next = [...prev, newAnn];
        next.sort((a, b) => a.startKey - b.startKey);
        return next;
      });
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [currentTime, annotations]);

  // Edit helpers
  const removeAnnotation = (id) => setAnnotations((p) => p.filter((a) => a.id !== id));
  const updateModifier = (id, val) => setAnnotations((p) => p.map((a) => a.id === id ? { ...a, modifier: val } : a));
  const updateDown = (id, val) => setAnnotations((p) => p.map((a) => a.id === id ? { ...a, down: val } : a));
  const updateTagName = (id, val) => {
    setAnnotations((prev) => {
      const t = prev.find((a) => a.id === id); if (!t) return prev;
      const duplicate = prev.some((a) => a.id !== id && a.startKey === t.startKey && normalize(a.tagName) === normalize(val));
      if (duplicate) return prev;
      return prev.map((a) => a.id === id ? { ...a, tagName: val } : a);
    });
  };

  // Exports (ALL annotations)
  const download = (filename, text, type = "application/octet-stream") => {
    const blob = new Blob([text], { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = filename;
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
  };

  const exportJSON = (rows = annotations) => {
    const payload = rows.map(({ startKey, endKey, tagName, modifier, down }) => ({
      TagName: tagName ?? "",
      StartTime: fmtMdotS(startKey),
      EndTime: endKey != null ? fmtMdotS(endKey) : "",
      Modifiers: modifier ?? "",
      Down: down ?? "",
    }));
    download(`annotations_${Date.now()}.json`, JSON.stringify(payload, null, 2), "application/json");
  };

  const toCSV = (rows) => {
    const esc = (v) => {
      const s = v == null ? "" : String(v);
      return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
    };
    const header = ["TagName","StartTime","EndTime","Modifiers","Down"];
    const lines = [header.join(",")];
    for (const r of rows) {
      const rowOut = [
        esc(r.tagName ?? ""),
        esc(fmtMdotS(r.startKey)),
        esc(r.endKey != null ? fmtMdotS(r.endKey) : ""),
        esc(r.modifier ?? ""),
        esc(r.down ?? ""),
      ];
      lines.push(rowOut.join(","));
    }
    return lines.join("\n");
  };
  const exportCSV = (rows = annotations) => download(`annotations_${Date.now()}.csv`, toCSV(rows), "text/csv");

  // Filters (search + tag filter)
  const [query, setQuery] = useState("");
  const [tagFilter, setTagFilter] = useState("All");

  const uniqueTags = useMemo(() => {
    const set = new Set(
      annotations.map((a) => a.tagName).filter((x) => (x ?? "").trim() !== "").map((x) => x.trim())
    );
    return ["All", ...Array.from(set).sort((a, b) => a.localeCompare(b))];
  }, [annotations]);

  const sorted = useMemo(() => [...annotations].sort((a, b) => a.startKey - b.startKey), [annotations]);

  const filtered = useMemo(() => {
    const q = (query ?? "").trim().toLowerCase();
    return sorted.filter((a) => {
      if (tagFilter !== "All" && (a.tagName || "").trim() !== tagFilter) return false;
      if (q) {
        const hay = `${a.startTime} ${a.endTime} ${a.tagName ?? ""} ${a.modifier ?? ""} ${a.down ?? ""}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });
  }, [sorted, tagFilter, query]);

  // Selection + bulk delete
  const [selected, setSelected] = useState(new Set());
  useEffect(() => { if (!hasSource) setSelected(new Set()); }, [hasSource]);
  const isSelected = (id) => selected.has(id);
  const toggleOne = (id) => setSelected((prev) => { const next = new Set(prev); next.has(id) ? next.delete(id) : next.add(id); return next; });
  const toggleAllFiltered = () => {
    setSelected((prev) => {
      const allIds = filtered.map((a) => a.id);
      const every = allIds.every((id) => prev.has(id));
      const next = new Set(prev);
      for (const id of allIds) every ? next.delete(id) : next.add(id);
      return next;
    });
  };
  const deleteSelected = () => {
    if (selected.size === 0) return;
    setAnnotations((prev) => prev.filter((a) => !selected.has(a.id)));
    setSelected(new Set());
  };

  const clearFilters = () => { setQuery(""); setTagFilter("All"); };

  // Pagination
  const PAGE_SIZE = 10;
  const [page, setPage] = useState(1);
  useEffect(() => { setPage(1); }, [query, tagFilter, annotations]);
  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const pageStart = (page - 1) * PAGE_SIZE;
  const pageRows = filtered.slice(pageStart, pageStart + PAGE_SIZE);
  const canPrev = page > 1;
  const canNext = page < totalPages;

  return (
    <section className="timeline">
      <h2>Timeline: Action Annotations</h2>

      {/* Filters */}
      <div className="timeline__filters" style={{ display: "grid", gap: 8, gridTemplateColumns: "1fr 220px auto", marginBottom:12 }}>
        <input type="text" placeholder="Search (start/end, tag, modifier, down)…" value={query} onChange={(e) => setQuery(e.target.value)} />
        <select value={tagFilter} onChange={(e) => setTagFilter(e.target.value)}>
          {uniqueTags.map((a) => (<option key={a} value={a}>{a}</option>))}
        </select>
        <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
          <button onClick={clearFilters} title="Clear search and filters">Clear</button>
        </div>
      </div>

      {/* Actions */}
      <div className="timeline__buttons" style={{ gap: 8, marginTop: 8 }}>
        <button onClick={toggleAllFiltered}>
          {filtered.length > 0 && filtered.every((a) => selected.has(a.id)) ? "Unselect All (Filtered)" : "Select All (Filtered)"}
        </button>
        <button className="btn--danger" onClick={deleteSelected} disabled={selected.size === 0}>Delete Selected</button>
        <div style={{ marginLeft: "auto", display: "flex", gap: 8 }}>
          <button className="btn--export" onClick={() => exportJSON()} title="Export all as JSON" disabled={filtered.length === 0}>Export JSON</button>
          <button className="btn--export" onClick={() => exportCSV()} title="Export all as CSV" disabled={filtered.length === 0}>Export CSV</button>
        </div>
      </div>

      {/* Table */}
      {filtered.length === 0 ? (
        <p className="timeline__empty">No annotations match your filters.</p>
      ) : (
        <>
          <table className="timeline__table">
            <thead>
              <tr>
                <th style={{ width: 28 }}>
                  <input
                    type="checkbox"
                    onChange={toggleAllFiltered}
                    checked={filtered.length > 0 && filtered.every((a) => selected.has(a.id))}
                    aria-label="Select all filtered"
                  />
                </th>
                <th style={{ width: 110 }}>Start</th>
                <th style={{ width: 110 }}>End</th>
                <th style={{ width: 220 }}>Tag Name</th>
                <th>Modifier</th>
                <th style={{ width: 90 }}>Down</th>
                <th style={{ width: 70 }}></th>
              </tr>
            </thead>
            <tbody>
              {pageRows.map((ann) => (
                <tr key={ann.id}>
                  <td>
                    <input
                      type="checkbox"
                      checked={isSelected(ann.id)}
                      onChange={() => toggleOne(ann.id)}
                      aria-label={`Select ${ann.tagName || "Custom Tag"} at ${ann.startTime}`}
                    />
                  </td>
                  <td>{ann.startTime}</td>
                  <td>{ann.endTime}</td>
                  <td>
                    <input
                      type="text"
                      value={ann.tagName}
                      placeholder="Tag name…"
                      onChange={(e) => updateTagName(ann.id, e.target.value)}
                    />
                  </td>
                  <td>
                    <input
                      type="text"
                      value={ann.modifier}
                      placeholder="Modifier…"
                      onChange={(e) => updateModifier(ann.id, e.target.value)}
                    />
                  </td>
                  <td>
                    <input
                      type="text"
                      value={ann.down ?? ""}
                      placeholder="1-4"
                      onChange={(e) => updateDown(ann.id, e.target.value)}
                      inputMode="numeric"
                    />
                  </td>
                  <td>
                    <button
                      className="btn--danger"
                      onClick={() => removeAnnotation(ann.id)}
                      onKeyDown={(e) => { if (e.code === "Space") e.preventDefault(); }}
                      title="Delete annotation"
                      aria-label={`Delete ${ann.tagName || "Custom Tag"} at ${ann.startTime}`}
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {totalPages > 1 && (
            <div className="pager">
              <button
                onClick={() => canPrev && setPage(page - 1)}
                disabled={!canPrev}
                title="Previous page"
                aria-label="Previous page"
                className="pagerBtn"
              >
                <MdChevronLeft size={20} />
              </button>

              <span className="pagerLabel">{page} of {totalPages}</span>

              <button
                onClick={() => canNext && setPage(page + 1)}
                disabled={!canNext}
                title="Next page"
                aria-label="Next page"
                className="pagerBtn"
              >
                <MdChevronRight size={20} />
              </button>
            </div>
          )}
        </>
      )}
    </section>
  );
}
