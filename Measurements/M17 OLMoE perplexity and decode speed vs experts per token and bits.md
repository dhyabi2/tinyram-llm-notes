---
id: M17
type: measurement
status: done
links: [M17, R18, P04]
date: 2026-10-29
---
# OLMoE perplexity and decode speed vs experts per token and bits

llama-perplexity on OLMoE-1B-7B Q4_K_M, a 550 KB technical-English corpus assembled from the llama.cpp docs (51 files), 24 chunks x 512 tokens, 4 threads, --override-kv olmoe.expert_used_count (runs/bench/olmoe_ppl_topk.json, olmoe_speed_topk.json, olmoe_ppl_bits.json). Active experts 8 / 6 / 4 / 2: PPL 18.96 / 20.64 / 27.61 / 131.4 (+0% / +8.8% / +45.6% / +593%); decode speed 41.5 / 49.5 / 61.4 / 73.9 tok/s (x1.00 / 1.19 / 1.48 / 1.78). Bits, requantised from Q4_K_M with --allow-requantize (so UPPER bounds on the loss): Q8_0 +0.2%, Q3_K_S +10.4%, Q2_K +26.0%. Reading: reducing active experts buys speed at a steep quality price (top-6: 1.19x for +8.8% PPL, above the 5% kill line; top-4: 1.48x for +46%). Caveats: one small model with 64 fine-grained experts and top-8; the 125B (512 experts, top-10) may behave differently; one corpus; PPL is not task quality. Consequence: top-k reduction and sub-4-bit are high-risk levers; in the combination model they are now scored 3-4.
- see [[R18 Combining the speed levers- all combinations]]
- see [[P04 Quality and quantization loss unmeasured]]
