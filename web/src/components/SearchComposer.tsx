// web/src/components/SearchComposer.tsx
import { memo } from "react";
import type { FormEvent } from "react";
import { Loader2, Search } from "lucide-react";
import type { SearchDomainOption, SearchSourceProvider } from "../types";

const SOURCE_OPTIONS: Array<{
  value: SearchSourceProvider;
  label: string;
  disabled?: boolean;
}> = [
  { value: "auto", label: "Auto (internal + web)" },
  { value: "retrieval", label: "Local Retrieval" },
  { value: "google", label: "Google PSE", disabled: true },
  { value: "serpapi", label: "SerpAPI" },
  { value: "browser", label: "Browser Retrieval" },
  { value: "all", label: "All Active Sources" },
];

// One representative query per routing intent (search / chat / tool) so anyone
// can exercise the intent router in a single click. The chip shows the query
// itself — clearest for a test-case affordance. See the intent-routed spec.
const EXAMPLE_QUERIES: Array<{ intent: string; icon: string; query: string }> = [
  { intent: "search", icon: "🔍", query: "find docs on cross-encoder reranking" },
  { intent: "chat", icon: "💬", query: "explain how FAISS indexing works" },
  { intent: "tool", icon: "🛠", query: "chart the tradeoffs between BM25 and dense retrieval" },
];

interface SearchComposerProps {
  query: string;
  searchUrl: string;
  topK: number;
  sourceProvider: SearchSourceProvider;
  isLoading: boolean;
  // When false (default), the raw retrieval URL input is hidden — the backend
  // resolves the URL from the selected Source. Shown only in dev (?dev=1).
  showUrlField?: boolean;
  // When false (default), the Source picker is hidden — the backend uses "auto"
  // fan-out. Shown only in dev (?dev=1).
  showSourcePicker?: boolean;
  // The topic domain is a product control, not a dev affordance: it is always
  // rendered. Options come from GET /api/search-domains so the 17 identifiers
  // and their order live in one place.
  domain?: string;
  domainOptions?: SearchDomainOption[];
  onQueryChange: (value: string) => void;
  onSearchUrlChange: (value: string) => void;
  onTopKChange: (value: number) => void;
  onSourceProviderChange: (value: SearchSourceProvider) => void;
  onDomainChange: (value: string) => void;
  onSubmit: (event?: FormEvent) => void;
  onExampleSelect?: (value: string) => void;
}

const DEFAULT_DOMAIN_OPTIONS: SearchDomainOption[] = [
  { name: "general", description: "Broad or mixed-topic search; default" },
];

export const SearchComposer = memo(function SearchComposer({
  query,
  searchUrl,
  topK,
  sourceProvider,
  isLoading,
  showUrlField = false,
  showSourcePicker = false,
  domain = "general",
  domainOptions = DEFAULT_DOMAIN_OPTIONS,
  onQueryChange,
  onSearchUrlChange,
  onTopKChange,
  onSourceProviderChange,
  onDomainChange,
  onSubmit,
  onExampleSelect,
}: SearchComposerProps) {
  return (
    <form className="composer" onSubmit={onSubmit}>
      <textarea
        aria-label="Question"
        value={query}
        onChange={(e) => onQueryChange(e.target.value)}
        onKeyDown={(e) => {
          if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
            e.preventDefault();
            onSubmit();
          }
        }}
        placeholder="Ask about your indexed docs, web results, or retrieval server output"
        rows={4}
      />
      {!isLoading && (
        <div className="example-chips">
          {EXAMPLE_QUERIES.map((ex) => (
            <button
              key={ex.intent}
              type="button"
              className="example-chip"
              title={`${ex.intent} example`}
              onClick={() =>
                onExampleSelect ? onExampleSelect(ex.query) : onQueryChange(ex.query)
              }
            >
              <span aria-hidden="true">{ex.icon}</span> {ex.query}
            </button>
          ))}
        </div>
      )}
      <div className="composer-controls">
        {showSourcePicker && (
          <label>
            Source
            <select
              value={sourceProvider}
              onChange={(e) => onSourceProviderChange(e.currentTarget.value as SearchSourceProvider)}
            >
              {SOURCE_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value} disabled={opt.disabled}>
                  {opt.label}
                </option>
              ))}
            </select>
          </label>
        )}

        {showUrlField && (
          <label className="url-field">
            Retrieval URL
            <input value={searchUrl} onChange={(e) => onSearchUrlChange(e.target.value)} />
          </label>
        )}

        <label>
          Domain
          <select
            value={domain}
            onChange={(e) => onDomainChange(e.currentTarget.value)}
          >
            {domainOptions.map((opt) => (
              <option key={opt.name} value={opt.name} title={opt.description}>
                {opt.name.replace(/_/g, " ")}
              </option>
            ))}
          </select>
        </label>

        <label>
          Top K
          <input
            min={1} max={20} type="number" value={topK}
            onChange={(e) => onTopKChange(e.currentTarget.valueAsNumber)}
          />
        </label>

        <button type="submit" disabled={isLoading || !query.trim()}>
          {isLoading ? <Loader2 className="spin" size={18} /> : <Search size={18} />}
          <span>{isLoading ? "Searching" : "Search"}</span>
        </button>
      </div>
    </form>
  );
});
