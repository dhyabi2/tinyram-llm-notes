---
id: M33
type: measurement
status: done
links: [Long real-prompt run (160 tokens): pool, overlap, and anchored 6 experts]
date: 2026-10-30
---
# Cache-aware routing in the real runtime (selection bias toward resident experts)

Implementation: GGML_MOE_CACHE_BIAS=lambda; for single-token steps the router probabilities used only for choosing the top-k are multiplied by (1+lambda) for experts that are resident in the read pool (the mixing weights still use the unmodified probabilities). Qwen3-30B Q4_K_M, pool+overlap 1500 MB. Speed, 9 GB ballast, real prompt, 48 tokens, 3 rounds: no bias 2.175 tok/s (hit 59 percent); lambda 0.3 2.378 (1.09x, hit 63 percent); lambda 1.0 2.947 (1.35x, hit 72 percent); lambda 3.0 4.016 (1.85x, hit 81 percent). Quality, decode-mode perplexity (-b 1 so the pool and bias are active), 4 chunks of 512 tokens, error bars about 0.65: no bias 7.425 (misses 40,865); lambda 0.6 7.546 (+1.6 percent, misses 16,632); lambda 1.0 7.430 (+0.07 percent, misses 11,317); lambda 3.0 7.427 (+0.03 percent, misses 3,801, 91 percent fewer). Text at lambda 1.0 identical to the unbiased 48-token text. So selection bias toward cached experts cuts disk reads by 60-90 percent at near-zero perplexity cost on this corpus and is the largest single lever found (1.35-1.85x on top of the pool); consistent with the published cache-conditional-experts result (miss rate down 50+ percent, ppl +0.1-3). Caveats: one corpus (documentation text), 4 chunks; behaviour on code, reasoning and long outputs untested; a larger bias keeps the same experts and could reduce diversity; residency is deterministic given the sequence but depends on pool size.
