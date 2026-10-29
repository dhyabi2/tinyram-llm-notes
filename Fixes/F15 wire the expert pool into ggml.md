---
id: F15
type: fix
status: open
links: [P01,T07]
date: 2026-10-28
---
# wire the expert pool into ggml

The C pool (byte-capacity LRU, 4 KiB-aligned O_DIRECT reads) is tested but not wired in. Worth building only if prefetch or explicit large reads show a real gain.
