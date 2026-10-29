---
id: M12
type: measurement
status: done
links: [M11, F20, P03]
date: 2026-10-28
---
# Hot set transfers poorly across prompts

Three real 250-300 token generations on Qwen3-30B-A3B: database-index explanation (db), lighthouse story, Python interval-merge code. Pinned hot set learned on one prompt and scored on another, cache = 6% of experts: same prompt 0.37 / 0.42 / 0.35; cross prompt 0.05-0.21 (db->story 0.106, db->code 0.206, story->code 0.048, code->story 0.061). A set learned on db+story scored on code: 0.144 at 6%, 0.227 at 10%, 0.438 at 20%. At 20% cache: same prompt 0.70-0.81, cross 0.31-0.55. Raw traces: runs/bench/moe_trace_qwen3-30b*.txt. The earlier 0.345 was learned and scored on halves of one prompt, so it is a same-prompt number.
- see [[M11 Expert reuse on real routing (Qwen3-30B-A3B)]]
- see [[F20 mlock the hot expert set]]
- see [[P03 The 100x claim is unproven]]
