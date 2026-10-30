---
id: M28
type: measurement
status: done
links: [Real-prompt A/B on the Mac: pool size, overlap, and fewer experts]
date: 2026-10-30
---
# Quality of fewer experts (anchored vs standard renormalisation) and of layer skipping, Qwen3-30B

Perplexity on a 500 KB text corpus, 10 chunks of 512 tokens, Qwen3-30B-A3B Q4_K_M (router renormalises the selected weights). Baseline 8 experts 9.338. Standard renormalisation over the kept experts (override of the expert count): 6 experts 9.781 (+4.7%), 4 experts 12.220 (+30.9%). Anchored (select top 8, normalise over all 8, compute only the first k; the kept weights sum to less than 1; GGML_MOE_USE_K): 7 experts 9.456 (+1.3%), 6 experts 9.813 (+5.1%), 4 experts 11.067 (+18.5%). So anchoring does not help at 6 or 7 experts but removes about 40 percent of the loss at 4 experts. Layer skipping with no training: skipping whole layers 2,6,...,46 (25 percent) 20.405 (+118%); whole layers 4,12,...,44 (12.5 percent) 11.262 (+20.6%); skipping only the MoE block in the 25 percent layers 11.622 (+24.5%). Layer skipping costs far more quality than reducing experts per token and the published claim of a 3x gain at 67 percent skip is not usable without training. Samples are small (10 chunks, error bars about 0.6). Ideas closed: layer skipping (training-free) refuted; anchored renormalisation useful only for aggressive cuts.
