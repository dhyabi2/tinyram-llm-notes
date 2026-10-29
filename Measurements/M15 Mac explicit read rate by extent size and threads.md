---
id: M15
type: measurement
status: done
links: [M04, R15, P01]
date: 2026-10-29
---
# Mac explicit read rate by extent size and threads

tools/read_bench.py (tinyram-llm working copy) on the 14 GB Qwen3.8-27B file, random page-aligned extents, os.pread with F_NOCACHE, 5 s per configuration. GB/s at 1 / 4 / 8 threads: 0.9 MB 1.25 / 2.69 / 2.78; 2.7 MB (one whole expert) 1.99 / 3.11 / 3.12; 4.0 MB 2.36 / 3.42 / 3.44. Same extents through mmap page touch (single-threaded Python loop, one touch per 4 KiB page, cache state not controlled): 0.28 / 0.48 / 0.60. So explicit reads reach ~3 GB/s with 4 reads in flight; the earlier model rate of 2.38 GB/s (4 MiB, queue depth 1) is conservative. The three tensors of an expert read in parallel behave like the 0.9 MB x 4 threads row (2.7 GB/s). Caveats: idle machine, no compute running; the mmap comparison is not the llama.cpp multi-threaded fault path (0.34 GB/s measured separately).
- see [[M04 Mac disk read-size sweep]]
- see [[R15 Brainstorm engine over every problem and the speed levers]]
- see [[P01 mmap read amplification]]
