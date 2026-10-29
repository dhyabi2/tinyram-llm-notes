---
id: F07
type: fix
status: done
links: [P06,T05,R04]
date: 2026-10-28
---
# preflight checks

Checks run in ~2 s: OS/CPU flags, RAM, decimal size parse, disk headroom, write test, download tool, HTTP range, exact remote size, llama-bench flags. Running it caught a bug in its own plan: the range check did not follow Hugging Face's 302 redirect.
