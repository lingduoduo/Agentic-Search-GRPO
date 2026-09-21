# Support the Python this package says it supports

## Goal

`pyproject.toml` declares `requires-python = ">=3.10"`. Make that true, and
leave behind something that notices when it stops being true.

## The reported symptom

On a 3.10 interpreter the suite did not run at all:

```
ImportError: cannot import name 'UTC' from 'datetime'
Interrupted: 187 errors during collection
```

`datetime.UTC` arrived in Python 3.11. Two `src/` modules imported it, and
because both sit on widely-imported paths, the failure cascaded into 187 test
modules.

## Why nothing caught it

Every CI workflow pins one interpreter:

```
.github/workflows/ci.yml:19          python-version: "3.12"
.github/workflows/ci.yml:40          python-version: "3.12"
.github/workflows/eval-gate.yml:37   python-version: "3.12"
.github/workflows/eval-gate.yml:125  python-version: "3.12"
.github/workflows/eval-gate.yml:158  python-version: "3.12"
```

So the declared floor is 3.10 and the tested floor is 3.12. Anything between
those imports cleanly for everyone who runs CI and fails for anyone on the
advertised minimum. The repo does intend 3.10 support -- it ships a `tomli`
backport and a test asserting the backport is installed -- which makes the gap
a drift rather than a decision.

This is the same shape as the SSE buffering defect: a property the test
environment is structurally unable to falsify.

## What was actually broken

Fixing the import exposed two more, each a place where 3.11 or 3.12 changed
behaviour rather than merely adding a name. That progression is the point --
one blocker hid the others.

**1. `datetime.UTC` (3.11+).** `src/internal/servers/web/tool_approval.py`,
`src/agents/tool/tool_calling.py`, and three test modules. Replaced with
`timezone.utc`, which is the same object on every version -- 3.11 added `UTC`
as an alias for it, so identity assertions like `tzinfo is timezone.utc` keep
holding.

**2. `except TimeoutError` after `asyncio.wait_for`.** In 3.11
`asyncio.TimeoutError` *became* the builtin `TimeoutError`; on 3.10 they are
unrelated classes, so the builtin does not catch what `wait_for` raises and the
approval broker's expiry path fell through. Fixed at
`tool_approval.py:152` by catching `asyncio.TimeoutError`, which is correct on
both -- and is already what the other thirteen timeout handlers in `src/` do.

`src/internal/hooks/executor.py:227` also catches a bare `TimeoutError`, and
that one is *right*: it follows `urllib`, where the raised timeout genuinely is
the builtin. Left alone.

**3. Two float assertions encoding 3.12-only arithmetic.** CPython 3.12 gave
`sum()` Neumaier compensation.

- `test_the_kernel_uses_compensated_summation_like_the_code_it_replaced`
  asserts `naive != sum(values)`. Its own docstring says "on CPython 3.12+",
  then asserts it unconditionally. Now gated on the version it names.
- `test_every_preset_keeps_its_exact_breakdown` compares production against a
  test-local `reward_baseline()` with `==`. The two sum in different orders, so
  they differ in the last ulp (`-0.11166666666666665` vs
  `-0.11166666666666668`); 3.12's compensation hides it and 3.10 does not.

  Note what this means: the gap is between two implementations in one process,
  not between Python versions. 3.12 was masking a difference that was always
  there. Changed to `pytest.approx`, which the neighbouring
  `test_every_preset_keeps_its_scalar_total` already uses on the same value.
  The guard's purpose -- catching a change to the reward math -- survives a
  tolerance this tight.

## The guard

`tests/unit/test_python_version_floor.py` parses `requires-python`, then walks
`src/`, `tests/` and `examples/` with `ast` and fails on any `from X import Y`
where `Y` postdates the floor. It carries a guard-the-guard test pinning the
floor at 3.10, so that if someone raises `requires-python` the check relaxes
deliberately rather than silently.

`src/context/enums.py` is exempt: it already guards `StrEnum` behind
`try/except ImportError` with a fallback, which is the sanctioned pattern and
the one to copy.

## What the guard cannot do

It catches names that do not exist. It does not catch **behaviour** that
differs -- which is exactly what (2) and (3) were. Nothing short of running the
suite on the floor catches those.

So the recommendation this leaves open: **add 3.10 to the CI matrix.** That is
the only change that closes the class rather than the instance. It is not done
here because it widens CI cost and may surface dependency-resolution work on
3.10, and that is a decision about the project's support commitment rather than
a bug fix.

## Not in scope

On the reporting environment, 8 modules still fail to collect and 9 tests still
fail, all for missing packages -- `boto3`, `botocore`, `fastmcp` -- that
`requirements-unit-test.txt` declares and that conda env does not have
installed. `pip install -r requirements-unit-test.txt` resolves them. No code
change applies.

## Verification

4155 passed on 3.10 (every remaining failure attributable to the missing
packages above), 4327 passed on 3.12. Both interpreters run the same tree.
