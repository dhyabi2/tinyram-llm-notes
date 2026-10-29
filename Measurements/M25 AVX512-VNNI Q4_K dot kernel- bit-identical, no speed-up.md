---
id: M25
type: measurement
status: done
links: [Clean Mac A/B: mmap vs pool vs overlapped pool, and dense-weight lock]
date: 2026-10-29
---
# AVX512-VNNI Q4_K dot kernel- bit-identical, no speed-up

Angle sweep idea 1 (hardware): the Q4_K x Q8_K dot is 256-bit AVX2 although the CPU has AVX512-VNNI; a standalone microbenchmark suggested 1.79x on the kernel. Implemented the real kernel in ggml (GGML_Q4K_VNNI=1): two 32-byte groups per zmm, vpdpbusd, per-sub-block scale multiply, lanes folded so the integer sums equal the AVX2 path. Bit-identical: OLMoE Q4_K_M (requantised from Q8_0) greedy text identical and perplexity 13.9573 in both modes. Speed on OLMoE Q4_K_M in RAM, 4 threads, 3 interleaved pairs: with --repack 0 decode 39.2 / 37.7 / 38.0 (off) vs 37.1 / 36.7 / 36.9 (on), prefill 149/145/142 vs 144/140/143; default repack decode 43.0/41.4/43.0 vs 42.7/42.9/42.7. No gain: the microbenchmark omitted the scale multiply and the shuffles and measured streaming, but real decode is bound elsewhere (dequant/memory access pattern, barriers). Verdict: refuted on this CPU; the 1.55x bound in the sweep was wrong. Side facts: default repack decode (43 tok/s) is 12% faster than --repack 0 (38) when the model fits in RAM, but repack needs a resident copy so it does not combine with the pool; prefill is faster with --repack 0 (145 vs 100 tok/s) on this build.
