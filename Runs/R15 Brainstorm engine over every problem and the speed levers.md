---
id: R15
type: run
status: done
links: [R13, M15, P01]
date: 2026-10-29
---
# Brainstorm engine over every problem and the speed levers

Round 1: 151 problems (every work-list item that is not verified-OK) x single and deep mode = 302 runs, 0 failures. Round 2 (speed hunt): 12 levers x 5 machine classes x 2 objectives = 120 problems x 2 modes = 240 runs, 0 failures, with the measured numbers and the already-ruled-out ideas in every prompt and a demanded >=3x claim with arithmetic. Total ~$1.0 for 542 runs (single ~$0.002, deep ~$0.003; the engine is flash DeepSeek non-thinking, 6-7 model calls inside each run). Merged into families by key-word overlap: runs/brainstorm/TRICKS_problems.md (229 families) and TRICKS_speed.md (58 families). Strong convergence in round 2: the top idea (expert-major execution over a same-prefix tree of positions) came from 13 of 120 problems independently. But its stated 17x fewer expert bytes double-counts: the real traces give 29.1 of 64 experts for 8 tokens = 2.2x per token (M15). The engine also says itself that compute caps decode at 4.6 tok/s, so >3x needs fewer active params. Next: prior-art and arithmetic screen of the top families.
- see [[R13 Itemised work list and test catalog]]
- see [[P01 mmap read amplification]]
