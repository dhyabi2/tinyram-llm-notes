---
id: F04
type: fix
status: done
links: [P06,T03]
date: 2026-10-28
---
# fio-free disk benchmark

fio printed a note before its JSON and broke parsing twice. `tools/disk_bench.py` (F_NOCACHE on macOS, O_DIRECT on Linux) measures 128 KiB, 1 MiB and 4 MiB reads.
