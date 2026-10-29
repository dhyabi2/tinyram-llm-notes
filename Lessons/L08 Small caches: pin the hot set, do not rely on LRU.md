---
id: L08
type: lesson
status: info
links: [M11, P01, F15]
date: 2026-10-28
---
# Small caches: pin the hot set, do not rely on LRU

With a cache smaller than one token's expert accesses (384 here), LRU, and so the OS page cache, gets ~0 hits on cyclic layer-by-layer access, while a pinned hot set learned from earlier tokens gets 0.31-0.48 at 5-10% cache. So the gain over plain mmap at 8 GB RAM can exceed the disk-speed ratio: pinned hot set + explicit 4 MiB reads ~1.1 tok/s vs mmap ~0.1 tok/s = about 10x (estimate; hit rates mapped from a 30B model). Still nowhere near 100x.
- see [[M11 Expert reuse on real routing (Qwen3-30B-A3B)]]
- see [[P01 mmap read amplification]]
- see [[F15 wire the expert pool into ggml]]
