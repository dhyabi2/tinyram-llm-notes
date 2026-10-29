---
id: P02
type: problem
status: done
links: [F06,L01]
date: 2026-10-28
---
# RAM cap not strict

Linux ballast sized from total RAM left the page cache bigger than the target (4 GB target, 6.6 GB MemAvailable). On macOS there is no cgroup, no drop_caches. Fixed by sizing from available memory and verifying; Mac uses a wired mlock ballast.
