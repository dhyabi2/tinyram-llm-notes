---
id: P01
type: problem
status: partial
links: [F14,F15,M01,M02,M03,M04]
date: 2026-10-28
---
# mmap read amplification

Reading a sparse MoE GGUF through `-lm mmap` on a machine whose RAM is smaller than the file wastes disk bandwidth.
- Colab gpt-oss-20b (12.1 GB, 4 GB target): ~5.4 GB read per token vs ~2 GB active weights, 0.094 tok/s.
- VPS Qwen3-30B-A3B (18.6 GB, 5 GB cap): ~1.1 GB per token, ~3,700 faults per token, 0.118 tok/s.
- Mac disk does 1.5-2.4 GB/s with 1-4 MiB reads but mmap paging got ~340 MB/s ([[L05 Read size decides disk speed]]).
Status: cause measured; fix (explicit large reads) only partly tested.
