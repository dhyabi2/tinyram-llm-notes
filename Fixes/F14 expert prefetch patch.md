---
id: F14
type: fix
status: partial
links: [P01,T08,M05,M06,M07]
date: 2026-10-28
---
# expert prefetch patch

Patch in ggml_compute_forward_mul_mat_id. GGML_MOE_PREFETCH=1: madvise(WILLNEED) on each chosen expert's range; =2: all threads touch their share of the pages (queue depth nth). Mac contended A/B 1.045x (noise); quiet Mac runs invalid (cached); VPS cold A/B running.

Update: VPS cold A-B test (see [[M07 VPS prefetch cold A-B test]]): mode 1 (madvise) = 1.41x, mode 2 (parallel touch) = 1.007x. Unproven on the Mac's NVMe (cannot start cold without sudo purge). Next: repeat on Colab (fast disk, fresh runtime) or with `sudo purge` on the Mac; then decide on the full pool ([[F15 wire the expert pool into ggml]]).
