---
id: M32
type: measurement
status: done
links: [Quantised KV cache (q8_0) on the pool path]
date: 2026-10-30
---
# Long real-prompt run (160 tokens)- pool, overlap, and anchored 6 experts

Qwen3-30B Q4_K_M, 9 GB wired ballast, 160 generated tokens, real prompt, 3 interleaved rounds. Plain mmap 0.417 / 0.418 / 0.440 tok/s (median 0.418); pool+overlap 1500 MB 1.544 / 2.024 / 2.001 (median 2.001, 4.8x); pool+overlap with anchored 6 experts (GGML_MOE_USE_K=6) 2.195 / 3.050 / 3.230 (median 3.05, 7.3x; perplexity +5.1 percent). Pool hit rate for 8 experts over the 160 tokens: 59 percent (148,349 hits / 104,526 misses), the same as at 48 tokens: the working set of real text is larger than a 1500 MB cache, so a longer run does not raise the hit rate; the cache needs to be bigger (more RAM) to improve it. Output text for 8 experts identical (md5 a2b25efc) between plain mmap and the pool.
