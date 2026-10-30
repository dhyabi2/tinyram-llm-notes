---
id: M29
type: measurement
status: done
links: [Quality of fewer experts (anchored vs standard renormalisation) and of layer skipping, Qwen3-30B]
date: 2026-10-30
---
# Pool warm start and adaptive pool size on the Mac

Warm start (GGML_MOE_POOL_WARM: at exit save the resident experts, MRU first; at the first MoE op queue them for background loading): primed from a DIFFERENT prompt, then the real prompt, 9 GB ballast, 48 tokens, 3 rounds. Cache misses 32,278 warm vs 32,312 cold: no benefit (hot sets do not transfer across prompts); prefill 28.0 s vs 26.2 s (slightly slower, the preload competes for the disk); decode medians 1.90 vs 2.09 / 0.73 vs 1.66 tok/s with one round distorted by swapping. Verdict: refuted for cross-prompt warm start; same-prompt (repeated task) not measured. Adaptive pool size (GGML_MOE_POOL_ADAPT=1: every 16 ops compare available memory with a 1 GB floor, shrink by 1/8 under pressure, grow back when more than 2.5 GB is free): 9 GB ballast, max 3000 MB, 48 tokens, 3 rounds: fixed 3000 MB 0.168 / 0.216 / 0.199 tok/s (collapse, thrashing), fixed 1500 MB 1.746 / 1.645 / 1.843, adaptive max 3000 MB 1.865 / 1.376 / 0.927 (median 1.376). Adaptive avoids the collapse (6.9x better than the oversized fixed pool) but is noisier and about 0.8x of a well-chosen fixed size; auto-sizing from available RAM at start plus adaptive shrink is the safe default. Text identical (md5 28e43bc1) in all variants.
