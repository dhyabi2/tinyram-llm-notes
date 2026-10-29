---
id: L17
type: lesson
status: done
links: [Paper check of 228 published-method rows]
date: 2026-10-29
---
# Deep paper check from raw text

24 agents re-read all 228 catalog rows from raw full text (225 read, 3 blocked: P-0137, P-0164, P-0194). Of 967 catalog numbers checked, 72 were wrong or absent. Transfer to our regime: no 138, partly 79, yes 8. Key changes to our beliefs: 16 GB hit rate probably about 0.77 not 0.95 (public benchmark on the 125B-class model), so the 16 GB base path is about 3.2 not 4.2 tok/s; top-k quality gate not established (renormalisation anchoring recovers most loss on a 512-expert model; plain cut on Qwen3-30B loses about 30 AIME points); replay order does not inflate our simulators (21/21 exact, order effects under 0.3%); real prefill coverage OLMoE 52-93%, Qwen3-30B 57-68%; forced-token share about 12.7% not 44-61%; dispatch 1-7% confirmed; stock-mmap gap may be larger than our emulation. Public prior art already streams a 125B-class model from flash. Details and the list of our own notes to fix: Playbook/papers/DEEP_SUMMARY.md.
