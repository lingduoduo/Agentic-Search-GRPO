# Retract #623's claim, and put a real build in CI

## Goal

Correct a false rule that #623 merged into the test suite, and add the CI job
that would have caught the original defect — and would have caught #623's
mistake too.

## What #623 got wrong

#623 claimed: *docker does not descend into an excluded directory, so
`!data/corpus.jsonl` under a bare `data` exclusion is unreachable.* It "fixed" a
`.dockerignore` that already worked.

Measured properly — one form per `docker build --no-cache`, buildkit 29.2.1:

| `.dockerignore` | corpus in image |
|---|---|
| `data` (pre-#622) | **absent** |
| `data` + `!data/corpus.jsonl` (#622 merged this) | present |
| `data/*` + `!data/corpus.jsonl` (#623 changed to this) | present |

So #622's fix was already correct. Buildkit honours the negation; the rule #623
asserted applies to the legacy builder, not the one in use.

### How it got through

Three compounding mistakes, each worth naming because each has a general form:

1. **The first probe ran against a stale local `main`.** `git fetch` without a
   reset left the checkout at a commit predating #622, so the `.dockerignore`
   under test was the first row of that table — `data` with no negations. The
   observation was real; the attribution was not.
2. **A plausible rule was adopted without testing the case it was about.** The
   pruning rule explains the observation and is even true of an older builder.
   #622's actual form was never built.
3. **The mutation check confirmed the story.** The "corrected" helper implemented
   the false rule, and the test asserting it passed — because test and helper
   encoded the same assumption. A later attempt to test all three forms reported
   every one as failing, because the grep matched the `FAIL:` string inside the
   echoed `RUN` command that `--progress=plain` prints.

The through-line: every wrong step measured something adjacent to the claim.

## The change

**The helper goes back to last-match-wins**, which is what buildkit does, and
`test_dockerignore_semantics_match_measured_behaviour` parametrises the three
measured forms so the model is pinned to observation rather than to whichever
rule sounded right. The docstring records that the earlier rule came from a build
against the wrong checkout.

`.dockerignore` keeps `data/*`. Both negated forms work, so switching back would
be churn; its comment no longer asserts the retracted rule and says plainly that
the form is a preference and which form is genuinely broken.

**`docker/Dockerfile.contract` plus a `Docker build context` CI job.** The
contract asserts, against a real build: the corpus and registry are in the image,
nothing else from `data/` is, the editable install registers `src`, and `src`
resolves from a working directory other than `/app`.

It skips `requirements.txt` — torch, transformers, faiss and pyserini are minutes
and gigabytes, and none of these questions involve them — while still exercising
the repo's real `.dockerignore`, which docker applies to the build context
whichever Dockerfile `-f` names.

Validated both ways: it passes on this branch, and reverting `.dockerignore` to
the genuinely broken `data`-with-no-negations form fails it at the first step.

This reverses the recommendation in #622, which argued against a Docker job on
cost. The cost argument was about the *full* image; this is seconds, and two
merged mistakes are the evidence that reading these files is not enough.

## What is still not covered

Whether `requirements.txt` installs on linux and whether the app boots. A full
build was attempted locally and abandoned at the torch/CUDA phase after host free
space fell from 24Gi to 11Gi. On a CI runner with room, that job is still worth
adding — as a scheduled build rather than a per-PR tax.
