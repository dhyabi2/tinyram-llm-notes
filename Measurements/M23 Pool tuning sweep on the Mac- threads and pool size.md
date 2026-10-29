---
id: M23
type: measurement
status: done
links: [Explicit read pool vs plain mmap on the Mac at 9 GB ballast]
date: 2026-10-29
---
# Pool tuning sweep on the Mac- threads and pool size

Qwen3-30B Q4_K_M, no ballast, llama-bench 16 tokens, pool log on. Threads: 4 threads 2.98 tok/s, 2 threads 2.74, 8 threads (hyperthreads) 0.258 - an 11x collapse from oversubscribing the 4 physical cores (ggml barriers spin). Pool size: 3000 MB 2.98 tok/s (61% hit, 7050 MiB read for 17 tokens); 6000 MB 3.46-3.48 tok/s (66% hit, 5997 MiB read, about 353 MiB per token); 12000 MB 3.30 (same hits as 6000 in a 16-token run). At 3.1 GB/s a token's reads take about 0.11 s of the 0.29 s per token, the rest is compute and sync, so reads and compute are serial today; perfect overlap would bound the speed near 1/max(0.12, 0.11-0.17) = 6-8 tok/s. Always use -t equal to physical cores.
