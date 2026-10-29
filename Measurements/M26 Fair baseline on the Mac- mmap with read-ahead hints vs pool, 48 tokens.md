---
id: M26
type: measurement
status: done
links: [Clean Mac A/B: mmap vs pool vs overlapped pool, and dense-weight lock]
date: 2026-10-29
---
# Fair baseline on the Mac- mmap with read-ahead hints vs pool, 48 tokens

Red-team check of the 5x claim. Qwen3-30B Q4_K_M, 9 GB wired ballast (all rows fully wired), 48 tokens, 3 interleaved rounds, 4 threads. Plain mmap 0.555 / 0.634 / 0.599 tok/s (median 0.599); mmap + madvise(WILLNEED) on the chosen experts 0.648 / 0.693 / 0.554 (median 0.648, 1.08x, inside the noise); mmap + parallel page touch 0.397 / 0.258 / 0.315 (median 0.315, 0.53x: worse); explicit pool 2.476 / 2.221 / 2.18 (median 2.22, 3.7x); pool + overlap (2 I/O threads) 2.985 / 2.358 / 2.79 (median 2.79, 4.7x). So read-ahead hints on the mmap path do not close the gap on macOS; the gain comes from explicit whole-expert reads plus the LRU pool, and overlap adds about 25% at 48 tokens (16% at 16 tokens). The 5x holds against the fairer baseline, and survives a 3x longer run (5.2x at 16 tokens vs 4.7x at 48). Remaining caveats: macOS only, one model, ballast-emulated small RAM, pool starts cold each run (steady state is probably higher).
