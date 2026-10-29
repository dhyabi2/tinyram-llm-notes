# tinyram-llm notes

Research notes, measurements and reports behind [tinyram-llm](https://github.com/dhyabi2/tinyram-llm): running very large sparse mixture-of-experts models from disk on machines with little RAM.

Plain Markdown (an Obsidian vault works too: open this folder as a vault). Everything is dated 2026-10-28/29; numbers are as measured then, with their conditions. Estimates and projections are labelled as such.

## Where to start
- `Playbook/COMPARISON.md`: measured with vs without the explicit expert read pool, plus projections.
- `Playbook/CLUSTER_PLAN.md`: does adding cheap nodes scale single-stream speed? (model + agent research; estimates only)
- `Playbook/IDEA_REGISTER.md` (+ `.csv`): every idea considered (80 canonical, from 349 raw), with status, evidence and next step.
- `Playbook/ANGLE_SWEEP.md`: eight-angle search for gaps between hardware limits and achieved speed, with skeptic checks.
- `Playbook/COMBINATIONS_TRACE.md`: trace-driven combinations of speed levers.
- `Playbook/pubtech/`: simulations of 48 published-idea families (A-F).
- `Playbook/papers/`: checks of 228 papers behind them (`SUMMARY.md`, `DEEP_SUMMARY.md`, per-paper `deep_P*.md`).

## Folders
- `Measurements/` M01-M26: every measured result with conditions (VPS, Colab, 16 GB Mac). `Fixes/`, `Lessons/`, `Problems/`, `Runs/`, `Tools/`: numbered notes that link to each other.
- `Playbook/*.json|csv`: raw data behind the reports.

## Caveats
- Measurements come from a 16 GB 2020 MacBook Pro, a small shared Linux VPS and Colab; the 125B-class model has never been run end to end. Read each note's conditions.
- Notes about unrelated internal tooling are not included, and local paths, addresses and account details were scrubbed.
- No license is granted for reuse of these notes beyond reading and linking. The code repository has its own license.
