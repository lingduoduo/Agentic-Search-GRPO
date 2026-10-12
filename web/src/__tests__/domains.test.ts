import { describe, expect, it } from "vitest";
import { MAX_SEARCH_DOMAINS, toggleDomain } from "../domains";

describe("toggleDomain", () => {
  it("replaces the web default instead of adding to it", () => {
    expect(toggleDomain(["web"], "financial")).toEqual(["financial"]);
  });
  it("adds up to the cap, then ignores more", () => {
    expect(toggleDomain(["financial"], "news")).toEqual(["financial", "news"]);
    const full = ["financial", "news", "code"];
    expect(full).toHaveLength(MAX_SEARCH_DOMAINS);
    expect(toggleDomain(full, "legal")).toEqual(full);
  });
  it("removes a selected category and falls back to web when empty", () => {
    expect(toggleDomain(["financial", "news"], "news")).toEqual(["financial"]);
    expect(toggleDomain(["financial"], "financial")).toEqual(["web"]);
  });
  it("selecting web explicitly clears the others", () => {
    expect(toggleDomain(["financial", "news"], "web")).toEqual(["web"]);
  });
});
