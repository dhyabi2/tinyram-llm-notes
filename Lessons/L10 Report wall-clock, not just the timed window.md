---
id: L10
type: lesson
status: info
links: [M07, R10]
date: 2026-10-28
---
# Report wall-clock, not just the timed window

A speedup measured over a short timed window can be 4x larger than the end-to-end gain when most of the run is load and prefill. Always record per-phase times (load, prefill, decode), wall-clock, and bytes at decode start and end, and state the ratio on both.
- see [[M07 VPS prefetch cold A-B test]]
- see [[R10 Playbook review of 22 methods]]
