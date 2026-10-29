---
id: R17
type: run
status: done
links: [R15, R16]
date: 2026-10-29
---
# Loop-until-dry brainstorm rounds 3-6 with an exclusion list

Four rounds (3-6) x 120 problems (12 levers x 10 outside-field lenses: databases, CPU microarchitecture, codecs, operating systems, cache theory, information theory, networking, emerging hardware, model structure, usage patterns) x single+deep = 960 runs, 0 failures, ~$2.6 (balance 8.36 -> 5.76 with the earlier rounds). Each round was told the 30 strongest idea families found so far and what is already ruled out. Share of runs not similar (word-overlap Jaccard < 0.30) to anything earlier: 97%, 80%, 82%, 73%; new families 116, 113, 114, 109 (452 in total; merged 307 families in runs/brainstorm/TRICKS_new.md, families_new.json). The loop stopped at the round limit, not because it ran dry. A prompt-truncation bug (6500-char cut dropping the ask and the lens) was found and fixed before the first real round. Most-repeated new ideas: prefill co-processor (21 problems), computational-storage FPGA (18), 2-bit VQ codes as LUT indices (18), persistent KV prefix replay (17, engine rating 72), delta coding between tokens (17), shared low-rank expert basis + residuals (16), trained per-layer router / early exit (16), expert-major prefill (16). Word-overlap novelty overstates real novelty (a new lens changes vocabulary); the prior-art screen decides.
- see [[R15 Brainstorm engine over every problem and the speed levers]]
- see [[R16 Screen of the top 22 brainstorm idea families]]
