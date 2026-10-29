---
id: M19
type: measurement
status: partial
links: [Naive mmap baseline under a strict memory cap (VPS)]
date: 2026-10-29
---
# Mac naive baseline under a wired ballast

Qwen3-30B (18.6 GB) plain mmap on the 16 GB Mac, 8 tokens, 1 rep, wired ballast then mincore check. No ballast: 0.65 tok/s, 0.57 GiB paged per token, 28% of the file resident. 6 GB: 0.57 tok/s, 21% resident. 9 GB: 0.23 tok/s, 1.29 GiB per token, 9% resident (1.6 GiB). 11 GB: the ballast did not wire (0%), run invalid. Short runs and a warm start, so shape only; next: repeat 9-10.5 GB with 3 reps and a wired check.
