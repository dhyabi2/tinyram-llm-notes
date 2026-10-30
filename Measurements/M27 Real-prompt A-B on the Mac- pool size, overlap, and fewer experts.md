---
id: M27
type: measurement
status: done
links: [Fair baseline on the Mac- mmap with read-ahead hints vs pool, 48 tokens]
date: 2026-10-30
---
# Real-prompt A-B on the Mac- pool size, overlap, and fewer experts

Qwen3-30B Q4_K_M, 9 GB wired ballast (every round fully wired), llama-completion, a real 25-token prompt, 48 generated tokens, greedy, 3 interleaved rounds, decode tok/s from the eval timing. Output text identical (md5 28e43bc1) for plain mmap and every pool variant with 8 experts: identity holds on the real model. Plain mmap 0.473 tok/s (0.473/0.517/0.363); pool sync 1500 MB 1.948 (4.1x); pool+overlap 800 MB 1.496 (3.2x, 40 percent hit); 1500 MB 2.217 (4.7x, 59 percent hit); 3000 MB 0.259 (0.55x: the pool plus the ballast exceeded free memory and the machine thrashed, although the hit rate was 71 percent); plain mmap with 6 experts 0.659 (1.39x). Earlier run, pool+overlap 1500 MB: 8 experts 1.47 (1.08-1.82), 6 experts 2.54 (1.7x), 4 experts 3.51 (2.4x). Two lessons: (1) llama-bench decodes a synthetic input that reuses a small expert set, so its 2.3-2.8 tok/s overstates real text; with a real prompt the same configuration gives 1.9-2.2 tok/s (hit rate 59 percent). The ratio over plain mmap is similar (4.1-4.7x). (2) the pool must fit in the memory that is really free: a pool bigger than the free RAM is far worse than plain mmap; auto-sizing from available RAM minus dense weights minus a 10 percent reserve avoids it (with a 9 GB ballast it would pick about 256-500 MB). Fewer experts per token raise speed more than the trace simulation predicted (6 experts 1.7x, 4 experts 2.4x over 8) because both reads and compute fall; quality cost: 6 experts +4.7 percent perplexity, 4 experts +30.9 percent.
