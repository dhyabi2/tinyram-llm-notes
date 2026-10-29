---
id: F32
type: fix
status: done
links: [R13, M11]
date: 2026-10-29
---
# One shared trace loader

Top of the investigators' do-next list: every trace reader in the pipeline outputs mislabels layers on the real trace (leading stray blk.47 line, layer taken from i%48). Our tools/trace_tools.py parses by layer name and drops the leading partial token; make it the only loader and gate on the 600/614 overlap with pin_db_10.
- see [[R13 Itemised work list and test catalog]]
- see [[M11 Expert reuse on real routing (Qwen3-30B-A3B)]]

Done: tools/trace_tools.parse is the only loader (layer from the tensor name, leading partial token dropped, malformed lines counted and skipped; 30 malformed-input cases + brute-force equality on 3 real traces + 8 random traces). tools/make_pin.py builds pin lists on it and agrees with the VPS-test pin file (>=600 of 614 shared). Committed in tinyram-llm (4d6153c).

