---
id: M20
type: measurement
status: done
links: [Naive mmap baseline under a strict memory cap (VPS)]
date: 2026-10-29
---
# Balanced cold madvise A-B on the VPS, 40 tokens, null pair

Qwen3-30B Q4_K_M, real story prompt, llama-completion, greedy, 40 tokens, 2 threads, cold (drop_caches before each run), balanced rotation of 3 conditions x 5 reps, shared throttled VPS disk, build 338fce6 plus the prefetch patch. Medians: off 0.1772 tok/s gen, wall 702 s, prompt eval 181 s, 40.7 GiB read; madvise 0.2928 tok/s, wall 572 s, prompt eval 139 s, 33.2 GiB read; null pair (off again) 0.1775 tok/s, wall 702 s, so the harness shows no drift (0.2% difference). madvise vs off: 1.65x on generation, 1.23x on end-to-end wall, 1.30x on prompt eval, 18% fewer bytes read. Ranges do not overlap (madvise 0.287-0.302, off 0.175-0.180). This replaces the earlier 9-token 1.41x / 1.09x wall figure: the longer cold run shows a larger and more reliable gain, and prefill also benefits. Limits: one host and disk (throttled ~120 MiB/s), one prompt, n=5, build differs from the Mac. Settles the W-001/003/007/016/018/020/024/026/029/031/038/041/042/046/077/079/096/100/104/112 theme (windows, order, n, null pair) for the VPS; a second host is still open.
