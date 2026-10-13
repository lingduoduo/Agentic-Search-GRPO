// web/src/components/__tests__/SearchComposer.test.tsx
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { SearchComposer } from "../SearchComposer";

const defaultProps = {
  query: "",
  searchUrl: "http://localhost:8001",
  topK: 5,
  sourceProvider: "retrieval" as const,
  isLoading: false,
  onQueryChange: vi.fn(),
  onSearchUrlChange: vi.fn(),
  onTopKChange: vi.fn(),
  onSourceProviderChange: vi.fn(),
  onDomainsChange: vi.fn(),
  onSubmit: vi.fn(),
};

describe("SearchComposer", () => {
  it("renders a textarea and submit button", () => {
    render(<SearchComposer {...defaultProps} />);
    expect(screen.getByRole("textbox", { name: /question/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /search/i })).toBeInTheDocument();
  });

  it("does NOT render a mode dropdown", () => {
    render(<SearchComposer {...defaultProps} />);
    expect(screen.queryByLabelText(/entry point/i)).not.toBeInTheDocument();
  });

  it("renders topK field", () => {
    render(<SearchComposer {...defaultProps} />);
    expect(screen.getByDisplayValue("5")).toBeInTheDocument();
  });

  it("hides the raw Retrieval URL field by default", () => {
    render(<SearchComposer {...defaultProps} />);
    expect(screen.queryByDisplayValue("http://localhost:8001")).not.toBeInTheDocument();
  });

  it("shows the Retrieval URL field when showUrlField is set (dev mode)", () => {
    render(<SearchComposer {...defaultProps} showUrlField />);
    expect(screen.getByDisplayValue("http://localhost:8001")).toBeInTheDocument();
  });

  it("disables submit when query is empty", () => {
    render(<SearchComposer {...defaultProps} query="" />);
    expect(screen.getByRole("button", { name: /search/i })).toBeDisabled();
  });

  it("enables submit when query has content", () => {
    render(<SearchComposer {...defaultProps} query="What is FAISS?" />);
    expect(screen.getByRole("button", { name: /search/i })).not.toBeDisabled();
  });

  it("disables submit while loading", () => {
    render(<SearchComposer {...defaultProps} query="hello" isLoading={true} />);
    expect(screen.getByRole("button")).toBeDisabled();
  });

  it("calls onSubmit when form is submitted", async () => {
    const onSubmit = vi.fn();
    render(<SearchComposer {...defaultProps} query="test" onSubmit={onSubmit} />);
    await userEvent.click(screen.getByRole("button", { name: /search/i }));
    expect(onSubmit).toHaveBeenCalledOnce();
  });

  it("submits on Cmd+Enter", async () => {
    const onSubmit = vi.fn();
    render(<SearchComposer {...defaultProps} query="hello" onSubmit={onSubmit} />);
    const textarea = screen.getByRole("textbox", { name: /question/i });
    await userEvent.click(textarea);
    await userEvent.keyboard("{Meta>}{Enter}{/Meta}");
    expect(onSubmit).toHaveBeenCalledOnce();
  });

  it("calls onQueryChange when user types", async () => {
    const onQueryChange = vi.fn();
    render(<SearchComposer {...defaultProps} onQueryChange={onQueryChange} />);
    await userEvent.type(screen.getByRole("textbox", { name: /question/i }), "hello");
    expect(onQueryChange).toHaveBeenCalled();
  });

  it("renders three example-query chips (one per intent)", () => {
    const { container } = render(<SearchComposer {...defaultProps} />);
    expect(container.querySelectorAll(".example-chip")).toHaveLength(3);
    expect(
      screen.getByRole("button", { name: /find docs on cross-encoder reranking/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /explain how FAISS indexing works/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /chart the tradeoffs between BM25 and dense retrieval/i }),
    ).toBeInTheDocument();
  });

  it("calls onExampleSelect with the example query when a chip is clicked", async () => {
    const onExampleSelect = vi.fn();
    render(<SearchComposer {...defaultProps} onExampleSelect={onExampleSelect} />);
    await userEvent.click(
      screen.getByRole("button", { name: /find docs on cross-encoder reranking/i }),
    );
    expect(onExampleSelect).toHaveBeenCalledWith("find docs on cross-encoder reranking");
  });

  it("falls back to onQueryChange when onExampleSelect is not provided", async () => {
    const onQueryChange = vi.fn();
    render(<SearchComposer {...defaultProps} onQueryChange={onQueryChange} />);
    await userEvent.click(
      screen.getByRole("button", { name: /explain how FAISS indexing works/i }),
    );
    expect(onQueryChange).toHaveBeenCalledWith("explain how FAISS indexing works");
  });

  it("hides example chips while loading", () => {
    const { container } = render(<SearchComposer {...defaultProps} isLoading={true} />);
    expect(container.querySelectorAll(".example-chip")).toHaveLength(0);
  });

  it("hides the Source dropdown by default", () => {
    render(<SearchComposer {...defaultProps} />);
    expect(screen.queryByText("Local Retrieval")).not.toBeInTheDocument();
  });

  it("shows the Source dropdown when showSourcePicker is set (dev mode)", () => {
    render(<SearchComposer {...defaultProps} showSourcePicker />);
    expect(screen.getByText("Local Retrieval")).toBeInTheDocument();
  });
});

describe("SearchComposer domain selector", () => {
  const domainProps = {
    ...defaultProps,
    domains: ["web"],
    domainOptions: [
      { name: "web", description: "Broad or mixed-topic search; default" },
      { name: "financial", description: "Markets, investments, banking" },
      { name: "social_media", description: "Public social posts" },
      { name: "news", description: "Current events" },
    ],
    onDomainsChange: vi.fn(),
  };

  it("renders the selector even without dev mode", () => {
    // Unlike Source and Retrieval URL, the domain is a product control.
    render(<SearchComposer {...domainProps} />);
    expect(screen.getByRole("group", { name: "Search source" })).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: "web" })).toBeChecked();
    expect(screen.queryByLabelText(/^Source$/)).not.toBeInTheDocument();
  });

  it("reports the selected domains", async () => {
    const onDomainsChange = vi.fn();
    render(<SearchComposer {...domainProps} onDomainsChange={onDomainsChange} />);
    await userEvent.click(screen.getByRole("checkbox", { name: "financial" }));
    expect(onDomainsChange).toHaveBeenCalledWith(["financial"]);
  });

  it("labels underscored identifiers readably", () => {
    render(<SearchComposer {...domainProps} />);
    expect(
      screen.getByRole("checkbox", { name: "social media" }),
    ).toBeInTheDocument();
  });

  it("keeps selected options enabled and shows the limit hint", () => {
    render(
      <SearchComposer
        {...domainProps}
        domains={["financial", "news", "social_media"]}
      />,
    );
    expect(screen.getByRole("checkbox", { name: "news" })).toBeEnabled();
    expect(screen.getByText("Up to 3 search sources")).toBeInTheDocument();
  });

  it("disables unselected real categories at the limit, with a description", () => {
    render(
      <SearchComposer
        {...domainProps}
        domains={["financial", "news", "social_media"]}
        domainOptions={[...domainProps.domainOptions, { name: "code", description: "Code" }]}
      />,
    );
    const code = screen.getByRole("checkbox", { name: "code" });
    expect(code).toBeDisabled();
    expect(code).toHaveAccessibleDescription(/Up to 3 search sources/);
  });

  it("lets web reset the selection at the limit", async () => {
    const onDomainsChange = vi.fn();
    render(
      <SearchComposer
        {...domainProps}
        domains={["financial", "news", "social_media"]}
        onDomainsChange={onDomainsChange}
      />,
    );
    await userEvent.click(screen.getByRole("checkbox", { name: "web" }));
    expect(onDomainsChange).toHaveBeenCalledWith(["web"]);
  });

  it("lists Auto first, reports it, and never limit-disables it", async () => {
    const onDomainsChange = vi.fn();
    const { rerender } = render(
      <SearchComposer {...domainProps} onDomainsChange={onDomainsChange} />,
    );
    const group = screen.getByRole("group", { name: "Search source" });
    expect(within(group).getAllByRole("checkbox")[0]).toHaveAccessibleName("Auto");
    await userEvent.click(screen.getByRole("checkbox", { name: "Auto" }));
    expect(onDomainsChange).toHaveBeenCalledWith(["auto"]);
    rerender(
      <SearchComposer
        {...domainProps}
        domains={["financial", "news", "social_media"]}
        onDomainsChange={onDomainsChange}
      />,
    );
    expect(screen.getByRole("checkbox", { name: "Auto" })).toBeEnabled();
  });
});
