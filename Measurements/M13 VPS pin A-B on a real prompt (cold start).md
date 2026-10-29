---
id: M13
type: measurement
status: done
links: [F20, F14, M07, M12, R10]
date: 2026-10-28
---
# VPS pin A-B on a real prompt (cold start)

Qwen3-30B-A3B Q4_K_M on the VPS (2 cores, 5.8 GB RAM, ~120 MiB/s disk), llama-completion with a real 35-token prompt (lighthouse story), 16 generated tokens, drop_caches before every run, 2 reps per condition, rotated order. Raw: runs/bench/ab-vps-pin.json. Hot lists: top 10% of (layer, expert) pairs from a trace; 'same' = built from this very prompt's own trace (IN-SAMPLE, an upper bound), 'cross' = built from a different prompt (database-index explanation). The pin (1.7 GB mlock) is loaded INSIDE the run.

| condition | generation tok/s (2 runs) | prompt phase | total read | wall |
|---|---|---|---|---|
| off | 0.1346 / 0.1332 | 182-184 s | 34.4-34.7 GiB | 593-599 s |
| madvise | 0.2119 / 0.2037 | 139-142 s | 29.7-29.8 GiB | 397 / 514 s |
| madvise + pin same-prompt 10% | 0.2522 / 0.2556 | 154-155 s | 29.8 GiB | 514-515 s |
| madvise + pin cross-prompt 10% | 0.1799 / 0.1803 | 162-163 s | 31.8 GiB | 548-549 s |

Generation speed vs off: madvise 1.55x, + same-prompt pin 1.90x, + cross-prompt pin 1.34x (worse than madvise alone, 1.55x). Wall-clock vs off: madvise 1.3x (median 456 s), same-prompt pin 1.16x, cross-prompt pin 1.09x. madvise also cut the PROMPT phase by ~24% and total bytes read by ~14%. Take-aways: (1) a hot list from another prompt HURTS (0.180 < 0.208): offline lists are not safe; (2) even the in-sample list costs ~60 s of load, so for 16 tokens it loses on wall time; pinning only pays on long generations; (3) the prompt phase is where most bytes are read (total read exceeds the 18.6 GB file). Limits: n=2 per condition, one host with a throttled disk, no output-identity check, no pinned-hit counter.
- see [[F20 mlock the hot expert set]]
- see [[F14 expert prefetch patch]]
- see [[M07 VPS prefetch cold A-B test]]
- see [[M12 Hot set transfers poorly across prompts]]
- see [[R10 Playbook review of 22 methods]]
