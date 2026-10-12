// web/src/components/SearchComposer.tsx
import { memo } from "react";
import type { FormEvent } from "react";
import { Loader2, Search } from "lucide-react";
import type { SearchDomainOption, SearchSourceProvider } from "../types";
import { AUTO, MAX_SEARCH_DOMAINS, toggleDomain } from "../domains";

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
  // The search source is a product control, not a dev affordance: it is always
  // rendered. Options come from GET /api/search-domains so the 20 categories
  // and their order live in one place.
  domains?: string[];
  domainOptions?: SearchDomainOption[];
  onQueryChange: (value: string) => void;
  onSearchUrlChange: (value: string) => void;
  onTopKChange: (value: number) => void;
  onSourceProviderChange: (value: SearchSourceProvider) => void;
  onDomainsChange: (value: string[]) => void;
  onSubmit: (event?: FormEvent) => void;
  onExampleSelect?: (value: string) => void;
}

const LIMIT_HINT_ID = "search-sources-limit";

const DEFAULT_DOMAIN_OPTIONS: SearchDomainOption[] = [
  { name: "web", description: "General information, websites, explanations, references" },
];

export const SearchComposer = memo(function SearchComposer({
  query,
  searchUrl,
  topK,
  sourceProvider,
  isLoading,
  showUrlField = false,
  showSourcePicker = false,
  domains = ["web"],
  domainOptions = DEFAULT_DOMAIN_OPTIONS,
  onQueryChange,
  onSearchUrlChange,
  onTopKChange,
  onSourceProviderChange,
  onDomainsChange,
  onSubmit,
  onExampleSelect,
}: SearchComposerProps) {
  const atLimit = domains.filter((d) => d !== "web").length >= MAX_SEARCH_DOMAINS;
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

        <fieldset className="search-sources">
          <legend>Search source</legend>
          <label title="Pick up to three sources automatically">
            <input
              type="checkbox"
              checked={domains.includes(AUTO)}
              onChange={() => onDomainsChange(toggleDomain(domains, AUTO))}
            />
            Auto
          </label>
          {domainOptions.map((opt) => {
            const checked = domains.includes(opt.name);
            // "web" stays clickable at the limit: it is how the user resets.
            const blocked = !checked && atLimit && opt.name !== "web";
            return (
              <label
                key={opt.name}
                title={[opt.description, opt.typical_sources?.join(", ")]
                  .filter(Boolean)
                  .join(" — ")}
              >
                <input
                  type="checkbox"
                  checked={checked}
                  disabled={blocked}
                  aria-describedby={blocked ? LIMIT_HINT_ID : undefined}
                  onChange={() => onDomainsChange(toggleDomain(domains, opt.name))}
                />
                {opt.name.replace(/_/g, " ")}
              </label>
            );
          })}
          <small id={LIMIT_HINT_ID} aria-live="polite">
            {atLimit ? `Up to ${MAX_SEARCH_DOMAINS} search sources` : ""}
          </small>
        </fieldset>

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
