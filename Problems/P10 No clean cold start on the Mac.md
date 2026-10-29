---
id: P10
type: problem
status: open
links: [F19,L01,M06]
date: 2026-10-28
---
# No clean cold start on the Mac

Cold cache needs `sudo purge` (password). madvise(DONTNEED) and reading another big file do not evict a cached file. A MoE larger than RAM (Qwen3-30B 18.6 GB) would stream naturally but its download is slow.
