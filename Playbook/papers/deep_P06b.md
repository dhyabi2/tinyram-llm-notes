# deep_P06b: raw-text re-check, rows P-0131, 0132, 0134, 0135, 0136, 0137, 0138

Scratch: <scratch> Full itemised entries (setup, numbers with where/ratio, catalog-number checks, transfer reason, quality, ablation, limitations) are in deep_P06b.json; this file lists the key points per row.

## P-0131 Deltafin blog (Michael Kerkhoff, Context Studios, 23 Sep 2026) - raw text read
- M5 Max 128 GB, 4 NVMe, Kimi K3 2.8T (1.45 TB experts, 16 experts/token). Decode 0.92/1.00 (512 tok, drafter off/on), 0.93/1.13 (128 tok), 0.96 on the 17-token prompt (median of 3); upstream M1 Max 0.2901 (0.68 on 17-token). TTFT 6.3 min for 512 tokens (prefill re-reads each layer's experts 8x; fix not built). SSD scaling 52/73/90/100 %: 1 to 2 drives = 1.40x only.
- Catalog numbers right. First pass wrong on "author unnamed" (byline is Michael Kerkhoff).
- Transfers: partly (same idea, but 128 GB, 4 SSD, Apple). Sub-linear SSD scaling is a caution on our ideal 2x-stripe arithmetic.

## P-0132 FluxMoE - raw text read (v1, v2, v3)
- v1 (Apr 2026): 4x L40, Xeon 8358, Qwen3-Next 80B 160 GB, Mixtral 94 GB. 3.0x/3.7x vs vLLM/vLLM-O at bs256, 63.9 % of vLLM at bs32, overhead 3.0 %: first-pass numbers correct for v1. Current v3 (10 Sep 2026, new author list) has 8x H20 GLM-4.5 355B: 7.2x vLLM, TPOT -79.0 %; Mixtral 4.3x KTransformers. Perplexity identical (Table 7).
- Correction: "storage hierarchy" = GPU compressed memory + host DRAM; no NVMe tier. Catalog wording overstated.
- Transfers: no (GPU, bs 32-256, frees VRAM for KV).

## P-0134 InstInfer - raw text read (v1)
- 11.1x vs FlexGen (OPT-13B, A6000, bs up to 256), 6.85x at bs64, 94.0 % KV-access reduction. New: only +4.6 % over best in-memory baseline (DeepSpeed bs16); CSD is an emulation (NVMeVirt, 8 ch x 1.4 GB/s) calibrated on Zynq7045; paper keeps QKV/O/FFN weight GeMM on the GPU because it is near CSD compute capacity; CSD 2-3 orders of magnitude weaker than a GPU (confirmed in text; first pass had it from a snippet).
- Transfers: no. Supports killing expert GEMV in CSD.

## P-0135 HillInfer - raw text read (v2 25 Mar 2026; v1 compared)
- 4.21-8.56x, latency -76.25..-88.32 %, batch 8, RTX 4090 + Xeon 8352V + 64 GB. Correction: SmartSSD peak bandwidth ~3.2 GB/s in v2 (first pass said ~4 GB/s from v1). Fig 5a: SmartSSD ~0.8 TFLOPS vs CPU ~40 TFLOPS. Catalog's "bandwidth remains the bottleneck" is an interpretation: the paper says the CSD bypasses host PCIe by P2P.
- Transfers: no. A SmartSSD's internal 3.2 GB/s equals one laptop NVMe, so no extra weight bandwidth.

## P-0136 Conduit - raw text read (full)
- Simulation (8 ch x 1.2 GB/s SSD, Xeon 5118, A100). LLM workloads are LLaMA2-7B INT8 inference/training, no MoE. 1.8x vs DM-Offloading, 1.8x vs GPU, 4.2x vs CPU; energy -46.8 % vs DM-Offl; 62 % of Ideal; 3.77 us overhead. Catalog and first pass correct. Transfers: no.

## P-0137 Samsung SmartSSD gen 2 - UNREAD
- BusinessWire returned Access Denied; Samsung newsroom fetch timed out; I did not bypass the block. No raw text. Nothing in the catalog line was verified.

## P-0138 DockerSSD - raw text read (v1)
- 7.9x = D-Cache vs H-Cache, analytic KV model added to an open simulator, 8 dense LLMs 137B-1T on 16-128 DockerSSD pools, seq 32K, batch 1/GPU; H-Cache swaps KV to SSD. In-SSD compute 1.7x slower than host (D-NoCache vs H-NoCache); D-Cache over H-Cache only up to 1.3x in the batch sweep. Up to 2.0x on I/O-intensive apps on the FPGA prototype (six RISC-V cores). Catalog correct; "did not open" now resolved. Transfers: no.

## Family: expert GEMV in computational-storage drives (screen 2 #1) and layer streaming (screen 2 #0)
Prior art read: InstInfer, HillInfer, Conduit, DockerSSD (above), plus read from search/arXiv this pass: 2508.06978 (SSD offload of MoE weights: 4.9x the per-token energy of HBM, 3.1x of CPU memory, up to ~12x; abstract/intro read), 2601.17063 FlashMoE (ML cache replacement, up to 51 % higher hit rate than LRU/LFU; abstract/intro read). From memory (unread): Cambricon-LLM, AiF (in-flash LLM), HILOS, LeoAM, InfiniGen. A search for in-storage expert computation found no paper that runs MoE expert GEMV inside a drive. What is new in our variant: nothing beyond two striped NVMe drives per our own test (pubtech/C.md); the nearest papers (InstInfer, HillInfer) offload only attention/KV and themselves argue weight GeMM overloads the CSD.
