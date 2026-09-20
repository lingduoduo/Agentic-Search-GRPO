# Documented settings must be settings

## Goal

Correct six environment variables that `configuration.md` documents but the
code does not read, and add the guard that would have caught them, so the table
cannot drift silently again.

## The problem

`configuration.md` documents 92 environment variables. Six of them do nothing
when set.

| Documented | Reality |
|---|---|
| `SEARCH_DIRECT_COS_MIN` | real, but the code reads `AGENTIC_SEARCH_SEARCH_DIRECT_COS_MIN` |
| `LATENCY_SLO_MS` (default `120`) | the gate is real, but it is driven by the CLI flags `--slo-ms` / `--qt-slo-ms` |
| `EF_SEARCH` | real, but the knob is the CLI flag `--hnsw_ef_search` at index-build time |
| `FAISS_INDEX_TYPE` (default `hnsw`) | real, but the flag is `--faiss_type`, the default is `Flat`, and it takes any `index_factory` string |
| `RERANKER_USE_ONNX` | no ONNX anywhere in the repo; the `ONNXReranker` it names does not exist |
| `BM25_VARIANT` | no BM25+ variant anywhere; nothing reads it |

Two of the six appear inside copy-pasteable startup commands in
`retrieval.md`, so a reader following our own documentation sets flags that
silently do nothing.

This is the shape #590 and #591 both found: an advertised capability with
nothing behind it. It is the most expensive kind of documentation error,
because it fails the way a correct setting would look if it happened not to
matter — no error, no warning, no log line. The operator concludes the knob
does not help, rather than that they never turned it.

Four of the six are worse than fiction: the capability is real and reachable
through a different interface. A reader who sets `EF_SEARCH` and measures no
change learns the wrong thing about HNSW, not about our docs.

## Architecture

Two halves: correct the table, then make it uncorrectable-by-accident.

**Correcting the table.** `SEARCH_DIRECT_COS_MIN` gains its
`AGENTIC_SEARCH_` prefix, in `configuration.md` and in the three places
`request-routing.md` names it. `LATENCY_SLO_MS`, `EF_SEARCH` and
`FAISS_INDEX_TYPE` leave the environment-variable table, because they are not
environment variables; each is described where its real CLI flag lives, with
`FAISS_INDEX_TYPE`'s wrong default corrected from `hnsw` to `Flat`.
`RERANKER_USE_ONNX` and `BM25_VARIANT` are removed outright, including from the
two commands in `retrieval.md` that set them.

**The guard.** Correcting six rows without a guard only resets the clock. This
drifted precisely because nothing checked, so the deliverable is a test:
every env var documented in `configuration.md` must be read somewhere in
`src/`, `examples/`, `tests/` or `.github/workflows/`.

The guard needs an escape hatch, because a variable can be legitimately
documented and legitimately absent from the code — one consumed only by a
container runtime, or by a third-party library reading its own configuration.
That hatch is an allowlist keyed by variable name whose value is the reason,
so an exemption has to be argued rather than merely added. An unexplained
entry is a review problem rather than an invisible one.

The guard matches how this repo already protects invariants: the
import-reachability guard over `servers/`, and the agent-menu invariant that
pins which tools are withheld.

## Testing

The guard is the deliverable, so it is mutation-checked: adding a fictional row
to `configuration.md` must turn it red, and removing that row must turn it
green again. A guard that cannot fail is worse than no guard — this session has
already produced two tests that passed before the change they claimed to
verify.

The guard is also checked against the state it was written for: on the
uncorrected docs it must name exactly the six variables above, no more and no
fewer. That is what proves it detects the thing it was built for rather than
merely passing afterwards.

## Limits

Documentation and one test. No production code changes, so no behavior changes.

The four variables backed by real capabilities are documented at their real
interface rather than wired to the documented names. Adding an env-var path to
`--slo-ms`, `--hnsw_ef_search` or `--faiss_type` would be a feature, needs its
own justification, and is not what a documentation-accuracy change should
smuggle in.

The guard checks that a documented variable is *read somewhere*. It cannot
check that the reading is meaningful — a variable read only to report itself,
the `ADAPTIVE_MMR` shape removed in #591, would still pass. That is a genuine
limit and is recorded here rather than papered over.

Out of scope: the other 86 documented variables' descriptions and defaults,
which this change does not audit; and env vars read by the code but absent from
the table, which is the opposite direction and a larger question.
