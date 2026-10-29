---
id: L14
type: lesson
status: info
links: [R15, R13, L02]
date: 2026-10-29
---
# Brainstorm multipliers collapse against the right baseline

Engine claims of 3x-100x were measured against plain mmap, an uncached baseline, or a double-counted ratio. Against our best path (explicit large reads + LRU) and the 4.57 tok/s compute ceiling they fall to 1.0-1.9x. Always recompute a claimed multiplier from our own traces and against the strongest baseline before it counts; a family produced independently by 10-13 problems is a sign the engine shares one bias, not that the idea is right.
- see [[R15 Brainstorm engine over every problem and the speed levers]]
- see [[R13 Itemised work list and test catalog]]
- see [[L02 Verifier scores are not stable]]
