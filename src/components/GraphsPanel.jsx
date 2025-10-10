import React from "react";

export default function GraphsPanel({ results, onClear }) {
  const hasData = results && typeof results === "object" && Object.keys(results).length > 0;

  const downloadJSON = () => {
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = `analysis_${Date.now()}.json`;
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
  };

  return (
    <section className="timeline" style={{ marginTop: 20 }}>
      <h2>Graphs</h2>

      {!hasData ? (
        <p className="timeline__empty">Run <strong>Analyze</strong> to see results here.</p>
      ) : (
        <>
          <div className="timeline__buttons" style={{ gap: 8, marginBottom: 12 }}>
            <button className="btn--export" onClick={downloadJSON}>Export JSON</button>
            <button className="btn--danger" onClick={onClear}>Clear Results</button>
          </div>

          {/* Simple textual preview until charts are implemented */}
          <div className="vp" style={{ padding: 16 }}>
            {Object.entries(results).map(([tag, arr]) => (
              <div key={tag} style={{ marginBottom: 10 }}>
                <div style={{ fontWeight: 600 }}>{tag}</div>
                <div className="mono" style={{ color: "var(--muted)" }}>
                  Counts: [{Array.isArray(arr) ? arr.join(", ") : ""}]
                </div>
              </div>
            ))}
          </div>

          {/* Placeholder for future charts */}
          <div className="timeline__empty" style={{ marginTop: 8 }}>
            {/* Charts coming after Analyze is implemented. */}
          </div>
        </>
      )}
    </section>
  );
}
