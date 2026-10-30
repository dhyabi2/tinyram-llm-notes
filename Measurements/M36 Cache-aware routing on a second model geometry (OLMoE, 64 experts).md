---
id: M36
type: measurement
status: done
links: [Best combination so far: pool + overlap + anchored 6 experts + cache bias]
date: 2026-10-30
---
# Cache-aware routing on a second model geometry (OLMoE, 64 experts)

OLMoE-1B-7B Q8_0 (16 layers, 64 experts, top-8, router weights not renormalised), 8 GB wired ballast, real prompt, 96 tokens, 3 rounds, pool 2000 MB (must exceed one token footprint: 16 layers x 8 experts x 3 tensors = 384 entries of 2.2 MB = 850 MB; an 800 MB pool sat below the cache cliff and showed no benefit from the bias, all primary accesses missed). Speed medians: plain mmap 2.098 tok/s; pool+overlap 5.297 (2.5x); bias 1.0 8.999 (4.3x); bias 3.0 13.024 (6.2x); misses 13,489 -> 6,582 -> 2,754. Quality, decode-mode perplexity, 4 chunks (baseline docs 13.989, code 7.336): bias 1.0 docs 14.049 (+0.4 percent), code 7.422 (+1.2 percent); bias 3.0 docs 16.383 (+17.1 percent), code 8.634 (+17.7 percent). So on this geometry bias 1.0 is nearly free and bias 3.0 is expensive; on Qwen3-30B (128 experts) bias 3.0 was cheap on documentation and +4.6 percent on code. Recommended default lambda = 1.0. Pool size requirement: the pool must hold well over one token footprint for the bias to work (misses are the only reads it can avoid).
