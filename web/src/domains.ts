// Mirrors MAX_SEARCH_DOMAINS in src/internal/tools/search.py.
export const MAX_SEARCH_DOMAINS = 3;

// Let the backend pick up to three categories for the query.
export const AUTO = "auto";

// One click on a "Search source" checkbox. "web" is the empty selection:
// choosing a real category replaces it, and clearing everything returns to it.
export function toggleDomain(current: string[], name: string): string[] {
  if (name === "web") return ["web"];
  if (name === AUTO) return current[0] === AUTO ? ["web"] : [AUTO];
  const real = current.filter((d) => d !== "web" && d !== AUTO);
  if (real.includes(name)) {
    const next = real.filter((d) => d !== name);
    return next.length ? next : ["web"];
  }
  if (real.length >= MAX_SEARCH_DOMAINS) return current;
  return [...real, name];
}
