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
  it("auto replaces any selection and toggles back to web", () => {
    expect(toggleDomain(["web"], "auto")).toEqual(["auto"]);
    expect(toggleDomain(["news", "code"], "auto")).toEqual(["auto"]);
    expect(toggleDomain(["auto"], "auto")).toEqual(["web"]);
  });
  it("a category replaces auto instead of appending", () => {
    expect(toggleDomain(["auto"], "news")).toEqual(["news"]);
  });
});
