---
id: L07
type: lesson
status: info
links: [M09, L01, P10]
date: 2026-10-28
---
# llama-bench decode reuses a small expert set

llama-bench decodes a fixed synthetic input, so the experts it routes to repeat from token to token. After the first cold pass the working set stayed in the page cache even with a verified 3 GB wired ballast and an 18.6 GB file: ~0 MiB per token and 8 tok/s. Per-token read figures from short cold runs mostly measure the cold start, so compare them only between runs with the same cold start. A real prompt would touch more experts.
- see [[M09 Mac Qwen3-30B-A3B prefetch test stayed cached]]
- see [[L01 A test that stays cached proves nothing]]
- see [[P10 No clean cold start on the Mac]]

**SUPERSEDED in part:** 'pin the hot set, do not rely on LRU' holds only when the cache is smaller than one token's footprint. At >=2 footprints LRU already hits ~0.55 and a share-based pinned set (0.26 at 4% share) is worse. Pinning is not the main lever; read size is.
