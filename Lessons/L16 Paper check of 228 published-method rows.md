---
id: L16
type: lesson
status: done
links: [Test program status after the parallel build-out]
date: 2026-10-29
---
# Paper check of 228 published-method rows

Twelve agents opened every catalog paper (219 of 228 pages opened; the fetch tool often returns a digest, not raw text) and searched for missed prior art. Catalog summaries: 127 accurate, 76 partly, 18 wrong, 7 unverified. Transfer to our regime (CPU-only, 8-32 GB, NVMe-streamed experts, batch 1): no for 89 rows, partly for about 45. Every headline speedup is GPU, phone, simulator or DRAM-resident. Outside confirmations of our measurements: mbolt co-activation reorder adds about 5% over prefetch (ours 1.045x); cross-layer prefetch about 1.0-1.08x in disk-bound regimes (three sources); static pin loses to LRU; ARC vs LRU under 0.01 hit; naive speculation under offload loses; lookup-table kernels lose to int8-dot on AVX2. Novelty threats: router bias for cache hits (2412.00099), SliceMoE, HOBBIT, FlashMoE, SpecMD, ExFold, llama.cpp discussion 23324, Flash-MoE. Challenges to check: LRU may fit expert access poorly (SpecMD, FlashMoE), recency-policy simulations can be inflated 27-29 percent by inconsistent replay (2608.07911), prefill may touch 80-98 percent of experts. Errors found in our own notes fixed (P-0025 misread in pubtech A). Wrong catalog rows listed in Playbook/papers/SUMMARY.md. Open: read SliceMoE and the Reddit 125B-class engine in full, and check our simulators for the replay-consistency issue.
