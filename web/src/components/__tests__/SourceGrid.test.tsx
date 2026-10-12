import { render, screen, fireEvent, act } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { SourceGrid } from "../SourceGrid";
import type { SourceDocumentView } from "../../types";

const doc: SourceDocumentView = {
  id: "D1",
  citation: "[D1]",
  title: "FAISS paper",
  content: "Dense retrieval with FAISS.",
  url: "https://example.test/faiss",
  score: 0.95,
  metadata: { source: "retrieval" },
};

const docNoUrl: SourceDocumentView = {
  ...doc,
  id: "D2",
  citation: "[D2]",
  title: "Local doc",
  url: null,
};

describe("SourceGrid", () => {
  it("renders empty state when documents list is empty", () => {
    render(<SourceGrid documents={[]} />);
    expect(screen.getByText(/no sources yet/i)).toBeInTheDocument();
  });

  it("renders document title", () => {
    render(<SourceGrid documents={[doc]} />);
    expect(screen.getByText("FAISS paper")).toBeInTheDocument();
  });

  it("renders a link when url is present", () => {
    render(<SourceGrid documents={[doc]} />);
    const link = screen.getByRole("link", { name: /faiss paper/i });
    expect(link).toHaveAttribute("href", "https://example.test/faiss");
    expect(link).toHaveAttribute("target", "_blank");
  });

  it("renders heading instead of link when url is absent", () => {
    render(<SourceGrid documents={[docNoUrl]} />);
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Local doc" })).toBeInTheDocument();
  });

  it("shows the categories that found a fused result", () => {
    const fused = { ...doc, metadata: { categories: ["financial", "news"] } };
    render(<SourceGrid documents={[fused]} />);
    expect(screen.getByText("financial · news")).toBeInTheDocument();
  });

  it("shows a single category", () => {
    const one = { ...doc, metadata: { categories: ["news"] } };
    render(<SourceGrid documents={[one]} />);
    expect(screen.getByText("news")).toBeInTheDocument();
  });

  it("shows nothing extra for missing or empty categories", () => {
    const empty = { ...doc, metadata: { categories: [] } };
    const { container } = render(<SourceGrid documents={[empty, doc]} />);
    expect(container.querySelector(".category-chips")).toBeNull();
  });

  it("renders the citation label", () => {
    render(<SourceGrid documents={[doc]} />);
    expect(screen.getByText("[D1]")).toBeInTheDocument();
  });

  it("renders multiple documents", () => {
    render(<SourceGrid documents={[doc, docNoUrl]} />);
    expect(screen.getByText("FAISS paper")).toBeInTheDocument();
    expect(screen.getByText("Local doc")).toBeInTheDocument();
  });

  it("shows mmr_rank badge from metadata", () => {
    const ranked = { ...doc, metadata: { source: "Local Retrieval", mmr_rank: 3 } };
    render(<SourceGrid documents={[ranked]} />);
    expect(screen.getByText("#3")).toBeInTheDocument();
  });

  it("applies green color to high score badge", () => {
    const { container } = render(<SourceGrid documents={[doc]} />); // doc.score = 0.95
    const badge = container.querySelector(".score-badge") as HTMLElement;
    expect(badge).not.toBeNull();
    expect(badge.style.color).toBe("rgb(34, 197, 94)");
  });

  it("adds id='source-[D1]' to each article", () => {
    render(<SourceGrid documents={[doc]} />);
    const article = document.querySelector('article[id="source-[D1]"]');
    expect(article).not.toBeNull();
  });

  it("content <p> has source-content--clamped class by default", () => {
    const { container } = render(<SourceGrid documents={[doc]} />);
    const p = container.querySelector("p");
    expect(p).not.toBeNull();
    expect(p!.className).toBe("source-content--clamped");
  });

  it("clicking 'show more' removes clamped class and updates button text", () => {
    const { container } = render(<SourceGrid documents={[doc]} />);
    const btn = screen.getByRole("button", { name: /show more/i });
    fireEvent.click(btn);
    const p = container.querySelector("p");
    expect(p!.className).toBe("");
    expect(screen.getByRole("button", { name: /show less/i })).toBeInTheDocument();
  });

  it("clicking 'show less' re-applies clamped class", () => {
    const { container } = render(<SourceGrid documents={[doc]} />);
    const showMore = screen.getByRole("button", { name: /show more/i });
    fireEvent.click(showMore);
    const showLess = screen.getByRole("button", { name: /show less/i });
    fireEvent.click(showLess);
    const p = container.querySelector("p");
    expect(p!.className).toBe("source-content--clamped");
  });

  describe("copy button", () => {
    beforeEach(() => {
      vi.useFakeTimers();
      Object.defineProperty(navigator, "clipboard", {
        value: { writeText: vi.fn().mockResolvedValue(undefined) },
        configurable: true,
      });
    });

    afterEach(() => {
      vi.useRealTimers();
      Object.defineProperty(navigator, "clipboard", {
        value: undefined,
        configurable: true,
      });
    });

    it("shows 'copied ✓' after clicking copy, then resets after 1.5s", async () => {
      render(<SourceGrid documents={[doc]} />);
      const copyBtn = screen.getByRole("button", { name: /copy/i });
      expect(copyBtn.textContent).toContain("copy");

      await act(async () => {
        fireEvent.click(copyBtn);
      });
      expect(copyBtn.textContent).toContain("copied ✓");

      act(() => {
        vi.advanceTimersByTime(1500);
      });
      expect(copyBtn.textContent).toContain("copy");
    });
  });
});
