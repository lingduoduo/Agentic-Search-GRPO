import { memo, useState, useCallback } from "react";
import { ExternalLink } from "lucide-react";
import type { SourceDocumentView } from "../types";

function scoreColor(score: number): string {
  if (score >= 0.7) return "rgb(34, 197, 94)";
  if (score >= 0.4) return "rgb(234, 179, 8)";
  if (score > 0)   return "rgb(249, 115, 22)";
  return "rgb(148, 163, 184)";
}

const SOURCE_COLORS: Record<string, string> = {
  "Browser Retrieval": "rgb(59, 130, 246)",
  "SerpAPI":           "rgb(139, 92, 246)",
  "Local Retrieval":   "rgb(107, 114, 128)",
  "All Active Sources":"rgb(14, 165, 233)",
};

interface SourceCardProps {
  document: SourceDocumentView;
  index: number;
}

const SourceCard = memo(function SourceCard({ document, index }: SourceCardProps) {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(document.content).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }, [document.content]);

  // Default to the app's local retrieval backend rather than "Unknown" when a
  // document arrives without a provider label (older records, edge paths).
  const source =
    typeof document.metadata.source === "string" && document.metadata.source
      ? document.metadata.source
      : "Local Retrieval";
  const mmrRank =
    typeof document.metadata.mmr_rank === "number"
      ? (document.metadata.mmr_rank as number)
      : index + 1;
  const sourceColor = SOURCE_COLORS[source] ?? "rgb(107, 114, 128)";

  return (
    <article
      className="source-card"
      id={`source-${document.citation}`}
    >
      <div className="source-meta">
        <span>{document.citation}</span>
        <span
          className="score-badge"
          style={{ color: scoreColor(document.score) }}
          title="Relevance score"
        >
          {document.score > 0 ? document.score.toFixed(3) : "—"}
        </span>
        <span
          style={{
            fontSize: "0.7rem",
            color: "rgb(148, 163, 184)",
            fontVariantNumeric: "tabular-nums",
          }}
          title="MMR rank"
        >
          #{mmrRank}
        </span>
        {Array.isArray(document.metadata.categories) &&
          document.metadata.categories.length > 0 && (
            <span className="category-chips" title="Found by these search sources">
              {(document.metadata.categories as string[]).join(" · ")}
            </span>
          )}
        <button
          className="source-copy-btn"
          onClick={handleCopy}
          title="Copy content"
          type="button"
        >
          {copied ? "copied ✓" : "⎘ copy"}
        </button>
      </div>

      <div className="source-tags">
        <span style={{ color: sourceColor, fontWeight: 600, fontSize: "0.7rem" }}>
          {source}
        </span>
      </div>

      {document.url ? (
        <a href={document.url} target="_blank" rel="noreferrer">
          {document.title}
          <ExternalLink size={14} />
        </a>
      ) : (
        <h3>{document.title}</h3>
      )}

      <p className={expanded ? undefined : "source-content--clamped"}>
        {document.content}
      </p>

      <button
        className="source-expand-btn"
        onClick={() => setExpanded((v) => !v)}
        type="button"
      >
        {expanded ? "show less ▴" : "show more ▾"}
      </button>
    </article>
  );
});

interface SourceGridProps {
  documents: SourceDocumentView[];
}

export const SourceGrid = memo(function SourceGrid({ documents }: SourceGridProps) {
  if (documents.length === 0) {
    return <div className="empty-state compact">No sources yet.</div>;
  }

  return (
    <div className="source-grid">
      {documents.map((document, idx) => (
        <SourceCard key={document.id} document={document} index={idx} />
      ))}
    </div>
  );
});
