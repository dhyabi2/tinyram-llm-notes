---
id: F20
type: fix
status: partial
links: [P01, R08, R09, M11]
date: 2026-10-28
---
# mlock the hot expert set

Next build, ranked first by plan E. Profile a real prompt with GGML_MOE_TRACE, rank (layer, expert) by frequency, then mlock their byte ranges (src0->data + e*nb[2]) inside the existing mmap; A/B on the VPS with drop_caches against 0.1464 (madvise on). Check ulimit -l and a 1 GB mlock probe first.
- see [[P01 mmap read amplification]]
- see [[R08 Plan E speedup steps ranked]]
- see [[R09 Plan F cache layout for the 125B]]
- see [[M11 Expert reuse on real routing (Qwen3-30B-A3B)]]

Refined after [[M12 Hot set transfers poorly across prompts]]: test two conditions on the VPS, same-prompt hot list (upper bound) and cross-prompt list (realistic offline), against madvise-only 0.1464. If the cross-prompt list does not help, the next version learns the set online from the prompt's own routing.

Result: see [[M13 VPS pin A-B on a real prompt (cold start)]]. An offline hot list from a different prompt makes decoding slower than madvise alone; an in-sample list helps decode (+22% over madvise) but its load cost exceeds the gain for 16 tokens. Do not ship a fixed hot list. Next: online learning from the prompt's own routing, a pinned-hit counter, and 64+ tokens (plan C3).
