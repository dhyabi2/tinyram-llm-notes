---
id: M22
type: measurement
status: done
links: [Mac naive baseline repeats at a file-to-cache ratio near 8x]
date: 2026-10-29
---
# Explicit read pool vs plain mmap on the Mac at 9 GB ballast

First measured 'with the tool' numbers. Qwen3-30B Q4_K_M, 16 GB Mac, 9 GB wired ballast (file about 8x the cache), llama-bench 8 tokens, 4 threads, 3 alternating pairs. Plain mmap: 0.243 / 0.340 / 0.421 tok/s, 0.68-1.14 GiB paged in per token. Explicit pread pool (1500 MB, F_NOCACHE, LRU): 1.383 / 1.719 / 2.019 tok/s, 0.06-0.07 GiB paged per token. Median ratio 1.72 / 0.34 = 5.1x (range 4.8-5.7x). The pool patch gives byte-identical output (4,225 identity checks). Short runs (8 tokens); pool starts cold each process; trend across pairs rising for both arms (OS state), so treat as 5x +- 1.
