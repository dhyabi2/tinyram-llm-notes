---
id: F01
type: fix
status: done
links: [P01,P06,T01]
date: 2026-10-28
---
# probe --repack 0

llama.cpp repacks weights into one large allocated buffer which defeats mmap on small RAM (`unable to allocate CPU_REPACK buffer`). Both probes pass `--repack 0`.
