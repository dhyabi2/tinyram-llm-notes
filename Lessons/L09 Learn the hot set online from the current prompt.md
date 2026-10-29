---
id: L09
type: lesson
status: info
links: [M12, M11, F20, R09]
date: 2026-10-28
---
# Learn the hot set online from the current prompt

Expert use is domain-specific: a hot list built offline from other prompts hits 0.05-0.21 at 6% cache, a list from the same prompt 0.35-0.42. So the pinned set must be learned online (decayed counts, or from the routing of the prompt's own tokens), not shipped as a fixed list. Plan F's table (hit 0.26 at 8 GB) assumed same-prompt learning; with an offline list expect about half of that.
- see [[M12 Hot set transfers poorly across prompts]]
- see [[M11 Expert reuse on real routing (Qwen3-30B-A3B)]]
- see [[F20 mlock the hot expert set]]
- see [[R09 Plan F cache layout for the 125B]]
