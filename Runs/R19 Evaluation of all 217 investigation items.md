---
id: R19
type: run
status: done
links: [R13, R16, R12]
date: 2026-10-29
---
# Evaluation of all 217 investigation items

29 agents judged each of the 121 optimistic items (O) and 96 new findings (N) on its own, opening the cited artifact and recomputing (Playbook/EVALUATION.md, 217 rows + detail, evaluation_raw.json). Verdicts: 130 confirmed, 80 partly confirmed (number or scope off), 3 already fixed, 4 unverifiable here, 0 refuted. Severity after the check: 40 high, 115 medium, 62 low. Needs: 92 local, 39 host run, 10 owner action, 4 hosted model, 72 nothing. In 19 items the investigator or the original output was wrong, e.g.: O-041 (the investigators' 6.27 footprints for 8 GB was wrong: the 8 GB machine has ~2.5 GiB of cache = 2.1 footprints, so our 4.1x serial / 7.0x overlap stand; only the label is ambiguous), O-072 (prefill compute of 109 s uses the decode rate; measured prefill ~310 GFLOP/s gives ~20 s, realistic TTFT 30-50 s), O-121 (jq exists), O-119 (fcntl 48 is F_NOCACHE), O-047/N-034 (MTP break-even alpha is ~0.74 not 0.5), N-046 (prefetch mode 1 does run in prefill). Biggest confirmed problems: the PHASE hook is on llama_decode which no tool calls (O-001/N-001); the wrapper, auditor and schema share no contract (O-005); the pre-registration lock and the warm-cache control are unimplemented and the PC control cannot exist on 5.8/12.7 GB hosts (O-016, O-011); the 1.15 kill line lies inside the expected 1.14-1.21 wall effect and n=6 has ~0.2 power (O-012, O-013); sim.py and cachemodel loaders shift layers because of the stray blk.47 first line (O-021, N-024); Mac sudo-free cold routes do not evict. Most high items need only text or code edits, not a host run.
- see [[R13 Itemised work list and test catalog]]
- see [[R16 Screen of the top 22 brainstorm idea families]]
- see [[R12 Twelve separate cluster sessions]]
