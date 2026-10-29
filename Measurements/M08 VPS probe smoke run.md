---
id: M08
type: measurement
status: info
links: [F06,F05]
date: 2026-10-28
---
# VPS probe smoke run

New probe on the VPS: disk sweep works (120 MiB/s at every size), tokens_counted 5, ballast resized once -> MEASURED-CAP (3.05 GB available vs 3.2 target). At a 3.2 GB cap the same model read ~4.6 GB/token vs ~1.1 GB at 5 GB (2-4 token run, a hint only).
