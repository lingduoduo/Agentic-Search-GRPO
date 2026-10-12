// Mirrors MAX_SEARCH_DOMAINS in src/internal/tools/search.py.
export const MAX_SEARCH_DOMAINS = 3;

// One click on a "Search source" checkbox. "web" is the empty selection:
// choosing a real category replaces it, and clearing everything returns to it.
export function toggleDomain(current: string[], name: string): string[] {
  if (name === "web") return ["web"];
  const real = current.filter((d) => d !== "web");
  if (real.includes(name)) {
    const next = real.filter((d) => d !== name);
    return next.length ? next : ["web"];
  }
  if (real.length >= MAX_SEARCH_DOMAINS) return current;
  return [...real, name];
}
