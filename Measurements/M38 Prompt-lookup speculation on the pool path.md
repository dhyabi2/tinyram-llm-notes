---
id: M38
type: measurement
status: done
links: [Generated text under cache bias (qualitative check)]
date: 2026-10-30
---
# Prompt-lookup speculation on the pool path

llama-lookup (n-gram drafting from the prompt), Qwen3-30B Q4_K_M, pool+overlap 1500 MB, 9 GB ballast, 64 tokens, 3 rounds, effective decode speed = tokens / target-pass time. Copy-heavy task (fix spelling in a given sentence): plain decode 2.01 / 2.01 / 2.06 tok/s; lookup draft 3 3.22 / 3.22 / 3.65 (1.6x; accept rate about 77 percent, 21 target passes for 64 tokens); draft 6 3.23 / 3.03 / 3.33 (1.6x). Open-ended explanation: plain 2.09; draft 3 2.15 (1.0x, range 1.43-2.42); draft 6 2.33 (1.1x). So prompt-lookup gives a real 1.6x only when the output copies the prompt (editing, summarising quotes, code refactoring) and nothing on free generation. Note: llama-lookup does not honour --no-warmup, so its load time was long (warmup pages every expert through mmap under the ballast, 130+ s); decode figures exclude it. Verdict: confirmed small niche lever; lossless.
