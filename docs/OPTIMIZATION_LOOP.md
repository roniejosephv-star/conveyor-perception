# Optimization Loop

**The Coach watches every Colab run and proposes the next code change.**

The full pipeline: `Colab notebook → GitHub Release → GitHub Action → PR → human review → merged → next Colab run`.

---

## Why

A solo founder can't manually review every session log + manually find the next improvement. The Coach agent does that. Each Colab run becomes a versioned artifact; the Action consumes it; Gemini suggests a code change; you review the PR. The next run picks up the change.

This is the **observe → orient → decide → act** loop, applied to a production CV pipeline. You're not waiting for a problem to surface — every run is a chance to improve.

---

## The 4 stages

```
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1 — Colab T4 (the data source)                            │
│                                                                   │
│  User opens notebooks/demo_v2.ipynb → clicks "Run all"            │
│  §1-§4 execute. §4 cell 15 publishes the run as:                  │
│  • Git tag: v0.0.{N} (next available semver)                      │
│  • Title: "Run v0.0.{N} — T4 {inference_ms}ms"                    │
│  • Asset: session.json (the full state.to_json() blob)            │
│  • Notes: T4 inference, errors, modules on/off, session ID        │
│                                                                   │
│  Requires: GITHUB_TOKEN Colab secret (classic PAT, scope: repo)  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2 — GitHub Action (.github/workflows/optimize.yml)         │
│                                                                   │
│  Trigger: release { types: [published] }                          │
│  Filter: only v0.0.* tags                                         │
│                                                                   │
│  1. Download current session.json from the release                │
│  2. Download previous session.json (gh release list --limit 2)   │
│  3. Run .github/workflows/coach_analyze.py                       │
│  4. If a suggestion was found, open a PR with peter-evans/...    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3 — Coach Analysis (coach_analyze.py)                      │
│                                                                   │
│  Inputs:                                                           │
│  • Current session log (full JSON: logs, errors, metrics, ...)   │
│  • Previous session log (for the diff)                            │
│                                                                   │
│  Process:                                                          │
│  • Builds a diff summary (metric changes, error deltas, toggles) │
│  • Calls Gemini 2.0 Flash with the diff + the rules               │
│  • Parses the JSON response: reason, file_path, old/new snippets │
│                                                                   │
│  Output:                                                           │
│  • /tmp/coach_suggestion.json (or NO_ACTION)                      │
│                                                                   │
│  Hard rules baked into the prompt:                                 │
│  • ONE change per run (quality over quantity)                     │
│  • No public API changes (don't rename Detector, etc.)            │
│  • No CI / Docker / harness changes (keep scope tight)            │
│  • If no metric changes AND no errors, respond "NO_ACTION"         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 4 — PR + human review (you)                                │
│                                                                   │
│  The Action opens a PR:                                            │
│  • Branch: coach/suggestion-v0.0.{N}                              │
│  • Title: "Coach suggestion: v0.0.{N}"                            │
│  • Body: Coach's reason + the suggested diff                      │
│                                                                   │
│  You review. Two outcomes:                                        │
│  • Merge → next Colab run picks up the change                      │
│  • Reject → Coach learns (next run's diff shows no improvement)    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Required setup (one-time)

### Colab side
- **GITHUB_TOKEN** Colab secret (classic PAT, scope: `repo`, 90-day expiry).
  - Get one at https://github.com/settings/tokens
  - In Colab: key icon → "+ Add new secret" → name: `GITHUB_TOKEN` → paste → toggle ON

### GitHub side
- **GEMINI_API_KEY** repo secret (for the Coach's Gemini call).
  - Get one at https://aistudio.google.com/app/apikey (free tier)
  - In GitHub: repo → Settings → Secrets and variables → Actions → New repository secret → name: `GEMINI_API_KEY` → paste value

The workflow uses the built-in `GITHUB_TOKEN` automatically (no PAT needed for the Action itself).

---

## What the Coach suggests

The Gemini prompt is constrained to:

- **Config tweaks**: batch size, imgsz, threshold, weight decay
- **Doc fixes**: typos in docstrings
- **Missing tests**: edge cases not covered
- **Refactors**: extract magic numbers into constants
- **New metrics**: things to track

The Coach is **blocked from**:
- Renaming public classes or functions
- Changing the MultitaskPipeline.step() signature
- Modifying CI / Docker / harness files
- Touching the conftest.py / test fixtures (those are mine to evolve)

This is enforced in the prompt, not by code. If Gemini ignores a rule, you reject the PR.

---

## How to debug when something breaks

| Symptom | Where to look | Fix |
|---|---|---|
| Colab publish cell fails | §4 cell 15 output | Check GITHUB_TOKEN is set; check token has `repo` scope |
| Action doesn't trigger | GitHub → Actions tab | Verify the tag starts with `v0.0.`; check the workflow file is at `.github/workflows/optimize.yml` |
| Action fails on download | Action logs → "Download current session log" step | Verify the release was actually published (not a draft) |
| Gemini call fails | Action logs → "Ask the Coach" step | Check GEMINI_API_KEY is valid; check Gemini free-tier quota |
| Coach returns invalid JSON | Action logs → "Check if Coach found a change" step | The prompt is too aggressive; tighten the rules or switch to a different Gemini model |
| PR has bad content | GitHub PR page | Reject + tweak the prompt |

---

## Why this matters for the interview

The optimization loop demonstrates 3 things EverestLabs cares about:

1. **Operational discipline.** Production CV systems need to know when they're drifting, when their accuracy is dropping, when their inference is slowing down. The Coach watches all of it.
2. **Closed-loop engineering.** Most ML projects are "train once, deploy, hope". This is "train, observe, improve, train again". That's how ROC teams scale.
3. **Self-documenting systems.** Every run is a versioned artifact. Six months from now, you can `git log v0.0.7..main` to see exactly what changed between runs, why, and what the metrics were.

The loop itself is ~100 lines of YAML + ~150 lines of Python. The leverage is enormous.

---

## Files

- `notebooks/build_demo_v2.py` — the publish cell (cell 15)
- `notebooks/colab_session.py` — `state.to_json()` produces the session blob
- `.github/workflows/optimize.yml` — the Action
- `.github/workflows/coach_analyze.py` — the Coach analysis script
- `tests/test_coach_analyze.py` — 10 tests for the analysis script
- `tests/test_demo_v2_builder.py` — 1 test for the publish cell (pinned structure)
- `docs/OPTIMIZATION_LOOP.md` — this file
