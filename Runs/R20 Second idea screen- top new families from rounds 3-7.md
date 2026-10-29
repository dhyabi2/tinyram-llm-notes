---
id: R20
type: run
status: done
links: [R16, R17, R15]
date: 2026-10-29
---
# Second idea screen- top new families from rounds 3-7

27 agents screened 26 families (20 from rounds 3-6, 6 from round 7); ranker in Playbook/idea_screen2_raw.json. Verdicts: 23 drop, 3 combine-with-other, 0 pursue. Novelty: 19 known, 7 partly known, 0 not found. Headline: none revolutionary. Only layer skipping reaches ~3x over our best path (3.0-3.2x at 16/32 GB with 67% of layers skipped) and it is published (Dr.LLM 2510.12773, LayerSkip, ShortGPT) with no quality support; realistic 1.1-1.4x. Others (vs our best path): multi-node expert-parallel decode (2 machines on a LAN; straggler 1.25-1.28x measured on traces; ~6.6 tok/s overlap bound; known OD-MoE, exo), top-k 10->4 (+ compute ceiling to ~6.0 tok/s, not 11.5: dense is 60% of FLOPs), shared low-rank basis (D2-MoE, MoE-SVD), intra-expert activation sparsity (TEAL, FloE, PowerInfer-2), vector quantisation with LUT matmul (T-MAC, AQLM), prompt-lookup speculation (llama.cpp ngram-cache), jump-forward decoding (XGrammar), 2-bit base + refinement tiles (HOBBIT). Errors found in brainstorm claims: prefill baseline used the decode rate (109 s vs ~19 s), 4 MiB rate applied to ~0.9 MB tensors, wrong baselines (mmap or serial instead of our best path), the 4.57 tok/s compute ceiling exceeded or ignored (claims up to 37 tok/s), dense FLOP share ignored, double-counted prefetch bytes, and the cross-layer predictor precision (0.55) is in-sample: held out it is 0.31-0.43.
- see [[R16 Screen of the top 22 brainstorm idea families]]
- see [[R17 Loop-until-dry brainstorm rounds 3-6 with an exclusion list]]
- see [[R15 Brainstorm engine over every problem and the speed levers]]
