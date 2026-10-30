---
id: M31
type: measurement
status: done
links: [Disk prompt cache for repeated prompts on the Mac]
date: 2026-10-30
---
# Quantised KV cache (q8_0) on the pool path

Qwen3-30B Q4_K_M, pool+overlap 1500 MB, 9 GB ballast, real prompt, 48 tokens, 3 interleaved rounds, flash attention on for q8_0. f16 KV 1.409 / 1.903 / 2.270 tok/s (median 1.903); q8_0 K and V 2.056 / 2.405 / 2.180 (median 2.18, 1.15x). Ranges overlap and the f16 runs rose across rounds, so the gain is within noise at this short context (KV is only about 20 MB); text differs as expected (md5 58acbbc0 vs 28e43bc1). Verdict: no measurable benefit at short context; only matters when long context competes with the expert cache for RAM.
