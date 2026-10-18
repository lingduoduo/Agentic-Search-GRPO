// Package render provides terminal output helpers for the enterprise CLI.
// Progressive() is extracted from cli/tui/viewport.go's appendToken()/finishAgent()
// pattern, adapted for non-TUI io.Writer use.
package render

import (
	"fmt"
	"io"
	"strings"
	"text/tabwriter"
	"time"

	"charm.land/glamour/v2"
	"charm.land/glamour/v2/styles"
	"github.com/lingduoduo/Agentic-Search/cli/models"
)

func urlStr(u *string) string {
	if u != nil && *u != "" {
		return *u
	}
	return "—"
}

func titleStr(t string) string {
	if t != "" {
		return t
	}
	return "—"
}

// Sources writes a tab-aligned source table to w. No-op if docs is empty.
func Sources(w io.Writer, docs []models.AgentDocument) {
	if len(docs) == 0 {
		return
	}
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "Cite\tTitle\tURL")
	fmt.Fprintln(tw, "────\t─────\t───")
	for _, d := range docs {
		fmt.Fprintf(tw, "%s\t%s\t%s\n", d.Citation, titleStr(d.Title), urlStr(d.URL))
	}
	_ = tw.Flush()
	fmt.Fprintln(w)
}

// newRenderer builds a glamour renderer matching tui/viewport.go's newMarkdownRenderer:
// dark style, zero left margin, word-wrap at width-4 columns.
func newRenderer(width int) (*glamour.TermRenderer, error) {
	style := styles.DarkStyleConfig
	zero := uint(0)
	style.Document.Margin = &zero
	return glamour.NewTermRenderer(
		glamour.WithStyles(style),
		glamour.WithWordWrap(width-4),
	)
}

// Progressive writes answer to w word-by-word, re-rendering with glamour after each word
// and using ANSI cursor controls to overwrite the previous render in place.
// wordsPerSecond controls speed (30 ≈ comfortable reading pace).
// width is the terminal column count for glamour word-wrap.
func Progressive(w io.Writer, answer string, wordsPerSecond float64, width int) {
	words := strings.Fields(answer)
	if len(words) == 0 {
		return
	}

	delay := time.Duration(float64(time.Second) / wordsPerSecond)
	r, err := newRenderer(width)
	if err != nil {
		// Fallback: plain progressive print without markdown.
		for _, word := range words {
			fmt.Fprintf(w, "%s ", word)
			time.Sleep(delay)
		}
		fmt.Fprintln(w)
		return
	}

	accumulated := ""
	prevLines := 0

	for _, word := range words {
		if accumulated != "" {
			accumulated += " "
		}
		accumulated += word

		rendered, err := r.Render(accumulated)
		if err != nil {
			rendered = accumulated
		}

		// Clear previous render: move up prevLines, erase to end of screen.
		if prevLines > 0 {
			fmt.Fprintf(w, "\033[%dA\033[J", prevLines)
		}

		fmt.Fprint(w, rendered)
		prevLines = strings.Count(rendered, "\n")
		time.Sleep(delay)
	}
}
