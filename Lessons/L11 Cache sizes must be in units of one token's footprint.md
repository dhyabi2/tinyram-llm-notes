---
id: L11
type: lesson
status: info
links: [M11, R07, R10]
date: 2026-10-28
---
# Cache sizes must be in units of one token's footprint

LRU behaviour is set by cache size relative to one token's expert accesses (30B: 384, 125B: 480), not by percent of all experts. A percent mapped from a 30B to a 125B model moves the cliff. Express hit rates against k = cache / footprint.
- see [[M11 Expert reuse on real routing (Qwen3-30B-A3B)]]
- see [[R07 Throughput table with measured inputs]]
- see [[R10 Playbook review of 22 methods]]
