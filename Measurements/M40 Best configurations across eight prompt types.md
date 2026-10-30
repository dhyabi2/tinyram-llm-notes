---
id: M40
type: measurement
status: done
links: [Best combination so far: pool + overlap + anchored 6 experts + cache bias]
date: 2026-10-30
---
# Best configurations across eight prompt types

Qwen3-30B Q4_K_M, 9 GB wired ballast, pool+overlap 1500 MB, 64 generated tokens, one run per prompt per variant, eight prompts (Q&A, code, math, story, JSON, translation, summary, plan). Decode tok/s medians (range): pool only 2.11 (2.10-2.45); bias 1.0 2.97 (2.72-3.51), 1.4x; 6 experts + bias 1.0 4.28 (3.33-4.52), 2.0x; 6 experts + bias 3.0 5.46 (4.23-6.37), 2.6x. The ordering held on every prompt; pool hit rate about 87 percent (the cache here is warm over 64 tokens) and misses cut as in the earlier runs. So the gains are not specific to the two prompts used earlier. Single runs per cell (no repeats), timing noise about 10 percent. Against plain mmap (0.42-0.47 tok/s) the best configurations are roughly 9-13x.
