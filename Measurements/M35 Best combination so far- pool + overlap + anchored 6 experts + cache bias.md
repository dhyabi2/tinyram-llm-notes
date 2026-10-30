---
id: M35
type: measurement
status: done
links: [Cache-aware routing on code text, and how large the bias can be]
date: 2026-10-30
---
# Best combination so far- pool + overlap + anchored 6 experts + cache bias

Qwen3-30B Q4_K_M on the 16 GB Mac, 9 GB wired ballast (file about 8x the cache), real prompt, 96 generated tokens, 3 interleaved rounds, pool+overlap 1500 MB. Speed medians: pool+overlap 8 experts, no bias 2.166 tok/s; anchored 6 experts + cache bias 1.0: 4.062 (1.88x; hit 75 percent); anchored 6 + bias 3.0: 5.837 (2.69x; hit 85 percent). Earlier plain mmap on 160 tokens 0.418, so the best configurations are about 9.7x and 14x plain mmap. Quality, decode-mode perplexity, 4 chunks of 512 tokens (baseline: documentation 7.425, code 4.053): anchored 6 alone docs 7.668 (+3.3 percent), code 4.267 (+5.3 percent); anchored 6 + bias 1.0 docs 7.688 (+3.5 percent), code 4.269 (+5.3 percent), misses cut 72-76 percent; anchored 6 + bias 3.0 docs 7.572 (+2.0 percent, within noise of the others), misses cut 90 percent; code with bias 3.0 + 6 experts not measured (bias 3.0 alone on code +4.6 percent). So once 6 experts are used the bias adds almost no extra quality cost and doubles to triples speed. Caveats: perplexity is a proxy (no task benchmarks), 4 chunks, error bars about 0.65 absolute so differences below 3 percent are not significant; Qwen3-30B-A3B only (the 125B has 512 experts and top-10, where the substitution freedom differs); speed measured with a ballast-emulated small-RAM Mac.
