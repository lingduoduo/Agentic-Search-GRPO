// Command memory manages a user's long-term memory via the Agentic Search
// backend /api/memory endpoints.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/lingduoduo/Agentic-Search/cli/api"
	"github.com/lingduoduo/Agentic-Search/cli/clientauth"
	"github.com/lingduoduo/Agentic-Search/cli/config"
	"github.com/lingduoduo/Agentic-Search/cli/exitcodes"
	"github.com/lingduoduo/Agentic-Search/cli/models"
)

func main() { os.Exit(run(os.Args[1:])) }

func usage() {
	fmt.Fprintln(os.Stderr, `memory: manage user memory via the Agentic Search backend

Usage:
  memory <command> [flags]

Commands:
  add <text...>            Save a memory
  list                     List stored memories
  search <query...>        Search memories (--top-k N)
  consolidate              Deduplicate, merge, resolve conflicts (--no-conflict skips conflicts)
  profile                  Show the user profile (--generate to rebuild via LLM)
  curate                   Reconcile memories from conversation (--session-id S)

Common flags: --url, --token, --user-id, --email, --secret, --json`)
}

func run(args []string) int {
	if len(args) == 0 {
		usage()
		return int(exitcodes.BadRequest)
	}
	cmd, rest := args[0], args[1:]

	fs := flag.NewFlagSet(cmd, flag.ContinueOnError)
	urlFlag := fs.String("url", "", "Backend URL (overrides AGENTIC_SEARCH_URL)")
	tokenFlag := fs.String("token", "", "Bearer token / JWT (overrides AGENTIC_SEARCH_PAT)")
	userIDFlag := fs.String("user-id", "", "User ID — mint a JWT when no token is given")
	emailFlag := fs.String("email", "", "Email embedded in the minted JWT")
	secretFlag := fs.String("secret", "", "JWT signing secret (else AGENTIC_SEARCH_AUTH_SECRET / AUTH_SECRET)")
	jsonFlag := fs.Bool("json", false, "output the raw response as JSON")
	topK := fs.Int("top-k", 5, "search: max results")
	noConflict := fs.Bool("no-conflict", false, "consolidate: skip conflict resolution (dedup, near-dup merge and summaries still run)")
	generate := fs.Bool("generate", false, "profile: rebuild via the LLM")
	sessionID := fs.String("session-id", "", "curate: restrict to one session")

	switch cmd {
	case "add", "list", "search", "consolidate", "profile", "curate":
	case "help", "-h", "--help":
		usage()
		return int(exitcodes.Success)
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q\n", cmd)
		usage()
		return int(exitcodes.BadRequest)
	}

	if err := fs.Parse(rest); err != nil {
		return int(exitcodes.BadRequest)
	}

	cfg := config.Load()
	if *urlFlag != "" {
		cfg.ServerURL = *urlFlag
	}
	// Token is optional: without one the backend uses the default user.
	if tok, err := clientauth.ResolveToken(*tokenFlag, cfg.APIKey, *userIDFlag, *emailFlag, *secretFlag); err == nil {
		cfg.APIKey = tok
	}
	client := api.NewClient(cfg)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	if err := dispatch(ctx, client, cmd, fs, *jsonFlag, *topK, *noConflict, *generate, *sessionID); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		var exitErr *exitcodes.ExitError
		if errors.As(err, &exitErr) {
			return int(exitErr.Code)
		}
		var apiErr *api.APIError
		if errors.As(err, &apiErr) {
			return int(exitcodes.ForHTTPStatus(apiErr.StatusCode))
		}
		return int(exitcodes.General)
	}
	return int(exitcodes.Success)
}

// printJSON writes v as indented JSON to stdout (used by every --json branch).
func printJSON(v any) error {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	fmt.Println(string(b))
	return nil
}

func dispatch(ctx context.Context, client *api.Client, cmd string, fs *flag.FlagSet, jsonOut bool, topK int, noConflict, generate bool, sessionID string) error {
	switch cmd {
	case "add":
		text := strings.TrimSpace(strings.Join(fs.Args(), " "))
		if text == "" {
			return exitcodes.Newf(exitcodes.BadRequest, "add requires memory text")
		}
		resp, err := client.SaveMemory(ctx, text)
		if err != nil {
			return err
		}
		if jsonOut {
			return printJSON(resp)
		}
		id := "(empty, not saved)"
		if resp.MemoryID != nil {
			id = *resp.MemoryID
		}
		fmt.Printf("saved: %s\n", id)
	case "list":
		resp, err := client.ListMemories(ctx)
		if err != nil {
			return err
		}
		if jsonOut {
			return printJSON(resp)
		}
		if len(resp.Memories) == 0 {
			fmt.Println("(no memories)")
		}
		for _, m := range resp.Memories {
			fmt.Printf("- (%s) %s\n", m.ID, m.Text)
		}
	case "search":
		query := strings.TrimSpace(strings.Join(fs.Args(), " "))
		if query == "" {
			return exitcodes.Newf(exitcodes.BadRequest, "search requires a query")
		}
		resp, err := client.SearchMemories(ctx, query, topK)
		if err != nil {
			return err
		}
		if jsonOut {
			return printJSON(resp)
		}
		if len(resp.Results) == 0 {
			fmt.Printf("no memories matched %q\n", query)
		}
		for _, r := range resp.Results {
			fmt.Printf("- [%.3f] (%s) %s\n", r.Score, r.ID, r.Text)
		}
	case "consolidate":
		resp, err := client.ConsolidateMemories(ctx, !noConflict)
		if err != nil {
			return err
		}
		if jsonOut {
			return printJSON(resp)
		}
		rep := resp.Report
		fmt.Printf("initial=%d duplicates_removed=%d conflicts_resolved=%d final=%d\n",
			rep.Initial, rep.DuplicatesRemoved, len(rep.ConflictsResolved), rep.Final)
		for _, c := range rep.ConflictsResolved {
			fmt.Printf("  conflict[%s]: kept %q, dropped %v\n", c.Attribute, c.Kept, c.Superseded)
		}
	case "profile":
		var resp *models.MemoryProfileResponse
		var err error
		if generate {
			resp, err = client.GenerateMemoryProfile(ctx)
		} else {
			resp, err = client.GetMemoryProfile(ctx)
		}
		if err != nil {
			return err
		}
		if jsonOut {
			return printJSON(resp)
		}
		if len(resp.Profile) == 0 {
			fmt.Println("(empty profile)")
		}
		for _, e := range resp.Profile {
			fmt.Printf("- %s / %s: %s\n", e.Topic, e.Subtopic, e.Content)
		}
	case "curate":
		var sid *string
		if sessionID != "" {
			sid = &sessionID
		}
		resp, err := client.CurateMemory(ctx, sid)
		if err != nil {
			return err
		}
		if jsonOut {
			return printJSON(resp)
		}
		if resp.Message != "" {
			fmt.Printf("status=%s: %s\n", resp.Status, resp.Message)
		} else {
			fmt.Printf("status=%s trajectory=%s counts=%v memories=%d\n",
				resp.Status, resp.TrajectoryID, resp.Counts, resp.MemoryCount)
		}
	}
	return nil
}
