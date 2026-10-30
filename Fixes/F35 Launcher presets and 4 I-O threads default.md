---
id: F35
type: fix
status: done
links: [M42 Tuning I-O threads and pool size for the best configuration]
date: 2026-10-30
---
# Launcher presets and 4 I-O threads default

Launcher got four presets from the measured results: safe (pool+overlap, identical output), balanced (cache bias 1.0), fast (bias 1.0 + 6 of 8 experts, ppl +3.5% docs / +5.3% code), fastest (bias 3.0 + 6 experts; bias 3 alone cost +17% on OLMoE, so check the model first). Default I/O threads now 4 on 4+ cores (64-token k6+b1: 1 thread 3.21, 2 threads 4.37, 4 threads 4.84 tok/s). Local commit fa26b9d in the code repo, not pushed. Tests 53/53 for the launcher, hygiene 223/223.
