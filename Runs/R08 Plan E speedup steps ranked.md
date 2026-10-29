---
id: R08
type: run
status: done
links: [P01, F14, F15, M07, M11]
date: 2026-10-28
---
# Plan E speedup steps ranked

15 stages, 70 KB, verifier 62 (62/62/62), 5 rubric items open. Ranking: (1) mlock a frequency-ranked hot set inside the existing mmap, about 60-100 lines, projected 0.1464/(1-0.345) = 0.22 tok/s on the VPS, stacks on madvise; kill if tok/s <= 0.1464, read/token not down 20%, or mlock fails (RLIMIT_MEMLOCK / macOS wired cap unknown). (2) Multi-MiB O_DIRECT/F_NOCACHE reads into the owned pool, 150-250 lines, up to 4.6-7x effective read rate, but ~1.0x on the VPS (flat 120 MiB/s) so provable only on a fresh Colab VM or a Mac with sudo purge; kill if Colab tok/s <= 1.2x mmap+madvise. (3) Layer-ahead prefetch: weak, <=1.2x, persistence 0.425 means blind prefetch wastes ~57% of the bytes. Optional: tiered experts (all Q2_K plus a Q4_K_M copy of the hottest 5%) = -29% expert bytes per token, gate on perplexity +5%. Key catch: on the VPS ~1.1 GB/token already equals the 120 MiB/s disk limit, so only reading fewer bytes can help there.
- see [[P01 mmap read amplification]]
- see [[F14 expert prefetch patch]]
- see [[F15 wire the expert pool into ggml]]
- see [[M07 VPS prefetch cold A-B test]]
- see [[M11 Expert reuse on real routing (Qwen3-30B-A3B)]]
