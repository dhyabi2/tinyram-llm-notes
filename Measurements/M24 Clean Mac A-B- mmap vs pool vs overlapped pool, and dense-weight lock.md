---
id: M24
type: measurement
status: done
links: [Explicit read pool vs plain mmap on the Mac at 9 GB ballast]
date: 2026-10-29
---
# Clean Mac A-B- mmap vs pool vs overlapped pool, and dense-weight lock

After closing Chrome (swap about 1 GB, memory 76% free) and gating on a fully wired ballast. Qwen3-30B Q4_K_M, 9 GB wired ballast, 4 threads, llama-bench, pool 1500 MB. 16 tokens, 3 interleaved triples, all 9 rows valid: plain mmap 0.476 / 0.513 / 0.521 tok/s (median 0.513, 0.52-0.57 GiB paged per token); synchronous pool 2.280 / 2.417 / 2.245 (median 2.28, 4.4x mmap, 0.01 GiB per token); pool with overlap (GGML_MOE_POOL_ASYNC=2: 2 blocking I/O threads, resident experts computed first, gate/up/down reads queued together) 2.777 / 2.645 / 2.507 (median 2.645, 5.2x mmap, 1.16x over the synchronous pool). The overlap gain is below the 1.25-1.5x bound derived in the angle sweep. Dense-weight lock (LLAMA_MLOCK_DENSE=1), 3 interleaved pairs with a clean machine: pool 2.00 / 1.61 / 1.98 vs pool+lock 1.98 / 2.12 / 1.50 tok/s, medians equal (1.98); paging per token near zero in both. So the earlier paging of dense weights was memory pressure from other apps, not a pool leak; the lock does nothing on a healthy machine (angle-sweep idea 'make dense weights resident' refuted in the clean state). Earlier runs with swap at 4-6 GB gave 0.04-0.10 tok/s and are invalid. Output identity of the overlap pool: 30/30 identical on the tiny model with tight capacities, sibling prefetch and fallbacks (stalls up to 500 per run).
