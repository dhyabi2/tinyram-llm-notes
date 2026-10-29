---
id: F33
type: fix
status: done
links: [R13, R16, T08]
date: 2026-10-29
---
# Phase timing hook on llama_context decode

Evaluation items O-001/N-001: the old PHASE patch hooked llama_decode(), which no tool calls at eec18f5 (they call llama_process -> llama_context::decode). New patch patches/llama-phase-timing.patch (31 lines, dry-run applies to the pristine file, committed 2732c0d in the private repo): with GGML_PHASE=1 every decode() prints 'PHASE decode n_tokens=N start= end=' on stderr; nothing is printed otherwise. Verified on the tiny model: one prompt call then single-token calls. tools/phase_parse.py turns the lines into warmup / prompt / decode / ttft / gap seconds (17 parser cases).
- see [[R13 Itemised work list and test catalog]]
- see [[R16 Screen of the top 22 brainstorm idea families]]
- see [[T08 moe-prefetch.patch]]
