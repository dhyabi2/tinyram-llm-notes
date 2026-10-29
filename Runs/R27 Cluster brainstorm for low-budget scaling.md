---
id: R27
type: run
status: done
links: [Master idea register built]
date: 2026-10-29
---
# Cluster brainstorm for low-budget scaling

16 agents (10 lenses, prior art, prices, model audit, synthesis, 2 skeptics). Verdict: adding one node does not reliably double single-stream speed. Only the first added node can approach 1.5-2x, and only if the dense part (57 percent of FLOPs) is also split across nodes (tensor parallel, about 96 exchanges per token) and the base node is I/O-bound (8 GB). Expert-only sharding saturates near 5 tok/s (Amdahl on dense). Layer split (llama.cpp RPC) 1.0-1.1x. Audited model, 16 GB nodes, 2.5GbE: about 4.1 / 6.2 / 9.5 / 11.2 tok/s at 1/2/4/6 nodes with the dense split (estimates; skeptic haircut 25-40 percent). Best value: 2-4 used 16 GB micro PCs or N305 mini PCs on 2.5GbE, 250-1100 dollars; needs custom runtime (replicated router, tensor-parallel attention, expert sharding by id, polling exchange); first test is a 4 KB ping-pong latency measurement between the Mac and one PC. Full report Playbook/CLUSTER_PLAN.md.05 USD so the brainstorm engine could not be used.
