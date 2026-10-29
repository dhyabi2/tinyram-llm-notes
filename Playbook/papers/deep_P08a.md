# deep_P08a: raw-text re-check of rows P-0167, P-0168, P-0169, P-0170, P-0172, P-0174, P-0175, P-0176

Date 2026-10-29. All eight papers read from pdftotext of the full arXiv PDF (scratch folder deep_P08a). Page numbers are PDF pages. Where a statement is from memory it is labelled so. Text treated as data. Our measurements are from Playbook/pubtech/D.md.

## P-0167 (family #6)

**Title/authors:** Dr.LLM: Dynamic Layer Routing in LLMs. Ahmed Heakl, Martin Gubri, Salman Khan, Sangdoo Yun, Seong Joon Oh (Parameter Lab, MBZUAI, NAVER AI Lab, Tuebingen). Header: 'Published as a conference paper at ICLR 2026'

**Read:** arXiv 2510.12773v2 (18 May 2026), pdftotext of full PDF, 1312 lines. (First pass read v1 which said 'Under submission'; v2 confirms ICLR 2026.) (raw_text_read=True)

**Setup:** Dense decoder LLMs: LLaMA-3.2-3B (Instruct, Base), LLaMA-3.1-8B (Instruct, Base), Qwen-2.5-3B-Instruct, Qwen-2.5-7B-Instruct (no MoE). One NVIDIA A100 40GB for router training (4 h, 25 epochs, routers 11M params for 3B = 0.14%, 16.8M for 8B = 0.56%). Router = Linear-GELU-Linear per layer (width 128) reading windowed mean-pooled hidden states (W=8); labels from offline length-aware MCTS (50 simulations, 961k forward passes) on 4,000 ARC/DART-Math examples; labels skip 7.6%, repeat 2.3%. IMPORTANT (Sec 3.1): routers run ONCE per input sequence, so the skip/execute/repeat path is fixed per prompt, not decided per generated token. Baselines: vanilla model, LayerSkip, ShortGPT, MindSkip, FlexiDepth (numbers copied from FlexiDepth paper, LLaMA-3-8B). Eval: lm-eval-harness, greedy, max 2048 tokens. Wall-clock hardware for the 29.21 s / Table 7 numbers is not stated beyond the A100 used for training (assumed A100, not stated). Batch sizes 1..128 in Table 7 (LLaMA-3B-Instruct).

**Headline numbers (what / value / baseline / ratio / where):**
- In-domain accuracy, all six models, ARC+DART 'Total' column: +0.9 to +3.2 points (e.g. LLaMA-3B-Instruct 46.1 -> 48.9; LLaMA-8B-Instruct 52.7 -> 54.7; Qwen-7B 59.4 -> 60.4) | baseline: vanilla model | ratio: n/a (points) | Table 3, p7
- Layers executed ('Num Layers' = SUM of layer applications over all forward passes of an example): LLaMA-3B-Instruct 331.1 -> 323.7 (-7.40); LLaMA-8B-Instruct 518.2 -> 509.6 (-8.66); LLaMA-3B-Base 815.4 -> 812.4 (-3.02); LLaMA-8B-Base 798.7 -> 794.5 (-4.22); Qwen-3B 832.4 -> 828.9 (-3.31); Qwen-7B 460.9 -> 457.5 (-3.39) | baseline: vanilla Num Layers | ratio: fraction of layer applications skipped, computed by me: 2.2%, 1.7%, 0.4%, 0.5%, 0.4%, 0.7% (mean ~1.0%) | Table 3, p7
- Out-of-domain average accuracy change, 4 instruct models, 8 benchmarks: -0.94 (LLaMA3B), -0.70 (LLaMA8B), -1.05 (Qwen3B), -0.70 (Qwen7B); mean -0.85; single-task drops up to -4.8 (Qwen3B AGIEval 54.2 -> 49.4), -3.7 (LLaMA3B PIQA 75.6 -> 71.9) | baseline: vanilla | ratio: n/a | Table 4, p7
- Wall time for 1000-token generation: 29.21 s | baseline: 34.49 s vanilla | ratio: 1.18x (15.3% less) | Sec 5 'Wall-clock latency', p6
- Throughput, LLaMA-3B-Instruct, batch 1: 31.78 tok/s | baseline: 25.05 tok/s | ratio: 1.269x (+26.89%) | Table 7, p10
- Throughput, batch 2/4/8/16/32/64/128: 41.87/76.12/115.56/122.89/130.83/108.05/113.38 | baseline: 37.97/70.74/107.74/117.86/125.43/103.18/109.12 | ratio: 1.10/1.08/1.07/1.04/1.04/1.05/1.04; average 92.56 vs 87.14 = +8.65% (mean of ratios 6.2%) | Table 7, p10
- Comparison on LLaMA3-8B, GSM8K/MMLU/HellaSwag/HumanEval average: Dr.LLM 67.4 (GSM8K 74.9, MMLU 66.8) | baseline: FlexiDepth 59.7, ShortGPT 48.9, MindSkip 48.2, LayerSkip 32.5 | ratio: n/a (FlexiDepth etc. copied from other paper; LayerSkip here has no LayerSkip training: GSM8K 0.4) | Table 5, p8

**Catalog and first-pass numbers checked:**
- Venue ICLR 2026 -> YES. v2 header 'Published as a conference paper at ICLR 2026' (first pass could not confirm because it read v1)
- Per-layer skip/execute/repeat routers on frozen model, ~4k MCTS labels -> YES. 4,000 MCTS-derived examples, Table 2
- About 5 layers skipped per example -> YES AS QUOTED, BUT MISLEADING. abstract/intro say 5.0 fewer layers per example on average, where a 'layer' count is cumulative over all forward passes (hundreds per example); real skipped share is ~1% of layer applications (Table 3), not 10-15%
- First-pass: 'only ~10-15% of layers skipped' -> NO. ~0.4-2.2% of layer applications per Table 3 (3-8.7 layers out of 331-832)
- First-pass: 'Table 3 accuracy +0.9..+3.2 with 3.0-8.7 fewer layers' -> YES. Total columns +0.9..+3.2 and -3.02..-8.66
- First-pass: 29.21 s vs 34.49 s, +26.89% batch 1, +8.65% batch average -> YES. as above; note 8.65% is the ratio of the Average row (92.56/87.14), which equals mean of throughputs not mean of per-batch ratios
- First-pass: routing F1 61% macro -> YES. Sec 5: macro F1 61%, per-layer accuracy 96.8%
- First-pass: 'abstract/body inconsistent on layers saved (5 vs up to 11)' -> PARTLY. the 11 is a per-dataset DART figure (-10.96 LLaMA-8B-Instruct); abstract 5 is the mean; also MCTS labels save only 1.82 layers on average (Sec 4.2)

**Transfers to our regime:** no. Assumption that breaks: (1) dense models and no expert reads, so skipping a layer saves compute not streamed bytes; (2) the skip is ~1% of layer applications and chosen once per prompt, so it cannot approach our 25-67% skip scenario; (3) the 15-27% wall-time gains are 10-25x larger than the 1-2% layer reduction and are not explained by the paper (undefined GPU timing setup), so they cannot be used as evidence for any byte-saving; (4) GPU, batch 1-128, A100; (5) routers are trained per model on 4k labels, retraining per MoE model needed. The skipped layers in a streamed-expert model would also skip that layer's expert reads, which the paper never studies.

**Quality loss:** In-domain (ARC/DART, trained on) +0.9..+3.2 points; out-of-domain mean -0.85 points over 8 benchmarks (Table 4), worst single tasks AGIEval -4.8 (Qwen3B), PIQA -3.7 (LLaMA3B), GPQA Diamond +2.6 (LLaMA3B). No perplexity reported.

**Ablation:** Fig 5 (3B): router bottleneck dim, number of linear layers, number of pooling windows; focal loss vs weighted CE (+1.1 ARC, +1.8 DART); Table 6: routing from previous-layer hidden state (74.5/38.6) vs from embeddings H1 (70.9/30.0) vs vanilla (73.5/35.2). No ablation separating skip from repeat on accuracy or speed.

**Limitations stated:** No dedicated Limitations section found (grep). Authors mention repeat only 2.3% of layers so batch sync cheap; 80-89% inter-sequence routing consensus (Table 8).

**Contradicts / supports our findings:** Partly contradicts a line in our D.md: 'ShortGPT/Dr.LLM remove or skip only 10-25% of layers ... our arithmetic gives 1.11-1.33x'. For Dr.LLM the true skip is ~1% of layer applications, so the matching claim does not hold for Dr.LLM (it holds only for ShortGPT-style 25% removal). Nothing in the paper contradicts our 1/kept arithmetic or the chance-level adjacent-layer Jaccard (different quantity).

## P-0168 (family #6)

**Title/authors:** LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding. Mostafa Elhoushi, Akshat Shrivastava, Diana Liskovich, Basil Hosmer, Bram Wasti, Liangzhen Lai, Anas Mahmoud, Bilge Acun, Saurabh Agrawal, Ahmed Roman, Ahmed A. Aly, Beidi Chen, Carole-Jean Wu (FAIR/GenAI/Reality Labs at Meta et al.)

**Read:** arXiv 2404.16710v4 (18 Oct 2024; 'Date: October 21, 2024'), full PDF text, 1794 lines. Venue 'ACL 2024' is NOT printed in the arXiv text (from memory only). First pass read only the abstract page; tables now read. (raw_text_read=True)

**Setup:** Dense Llama models: Llama2 7B/13B continual pretraining (Table 3), Llama3 8B and Llama3.2 1B continual pretraining (Table 2), Llama2 1.5B and 7B pretrained from scratch on 26B tokens (Table 4), Llama1 7B code finetune (Table 5), Llama 1.5B on TOPv2 (Table 6, 11). Training with layer dropout (higher rate for later layers) and an early-exit loss with shared LM head, on 32-128 A100 (Table 8). Inference: Table 3/4/6 on NVIDIA H100, Table 5 on A100, HuggingFace implementation, greedy, 512 max new tokens; self-speculative decoding = draft with first E layers, verify with remaining layers, shared KV cache. CPU: Table 11 small Llama-like 1.5B TOPv2, 7 speculations, 50 tokens. Baseline = autoregressive full model of the same (non-LayerSkip) training; also Draft&Verify numbers copied.

**Headline numbers (what / value / baseline / ratio / where):**
- Self-speculative speedup CNN/DM, Llama2 7B continual pretrain, E=8 d=12: 127.9 tok/s, speedup printed 1.86x | baseline: autoregressive 62.7 tok/s | ratio: ratio of tok/s columns 2.04x (paper's speedup is mean of per-sample time ratio, 1.86x) | Table 3, p16
- Same, Llama2 13B, E=15 d=12: 70.2 tok/s, 1.81x | baseline: 37.2 tok/s | ratio: 1.89x of columns | Table 3, p16
- XSUM Llama2 7B / 13B: 104.7 tok/s 1.54x / 60.5 tok/s 1.34x | baseline: 63.4 / 43.8 tok/s | ratio: 1.65x / 1.38x of columns; Draft&Verify 1.48x on 13B beats LayerSkip 1.34x | Table 3, p16
- HumanEval Llama2 7B / 13B: 122.8 tok/s 1.83x / 84.2 tok/s 1.66x | baseline: 62.9 / 48.9 tok/s | ratio: 1.95x / 1.72x of columns | Table 3, p16
- Pretrained from scratch, CNN/DM, Llama2 7B 26B tokens (the catalog's 2.16x): 145.6 tok/s, 2.16x, token acceptance 77.8% | baseline: 64.5 tok/s | ratio: 2.26x of columns | Table 4, p16
- Same, Llama2 1.5B: 167.4 tok/s, 1.76x | baseline: 91.6 tok/s | ratio: 1.83x | Table 4, p16
- Code finetune Llama1 7B, HumanEval E=6 d=12 (the catalog's 1.82x): 62 vs 34 (units tok/s) , 1.82x, acceptance 45% | baseline: autoregressive 34 | ratio: 1.82x | Table 5, p16
- TOPv2 Llama 1.5B (the catalog's 2.0x), exit layer E=6: 18 ms/token, EM 82.9%, acceptance 76.0%, 2.0x | baseline: 36 ms/token, EM 85.9% | ratio: 2.0x; E=12 22 ms 1.64x; E=18 29 ms 1.24x | Table 6, p16
- Early exit alone (no verification), Llama2 7B CNN/DM E=8: ROUGE-2 0.012, 232 tok/s | baseline: ROUGE-2 0.079 full model | ratio: quality collapses (0.15x of full ROUGE-2) so speed is not usable | Table 3, p16
- CPU, TOPv2 1.5B, self-speculation E=6 / 12 / 18: 87 / 104 / 134 ms per token | baseline: 165 ms autoregressive | ratio: 1.90x / 1.59x / 1.23x | Table 11, p26
- Last-layer quality cost of the training recipe, Llama3 8B continual pretrain: MMLU 60.5, GSM8K 45.0, HumanEval 28.7, MBPP 40.0 | baseline: baseline continual pretrain 66.5, 54.2, 37.8, 49.0 | ratio: -6.0, -9.2, -9.1, -9.0 points | Table 2, p12

**Catalog and first-pass numbers checked:**
- Layer dropout + early-exit loss at training time, with self-speculative decoding -> YES. Sec 4
- Speedups 2.16x on CNN/DM -> YES. Table 4: Llama2 7B pretrained from scratch, 26B tokens, H100
- 1.82x on coding -> YES. Table 5: Llama1 7B code finetune, A100
- 2.0x on TOPv2 -> YES. Table 6: 1.5B model, E=6; other exit layers give 1.24x and 1.64x
- 'needs c...' (needs custom training) -> YES. Limitations p25: requires finetuning or pretraining with the recipe, tuned hyperparameters
- First-pass: 'Llama of unstated sizes; hardware not stated' -> WRONG/ABSENT. sizes 1B-13B and H100/A100 are stated (Tables 2-6, Sec 6.2)
- Abstract says 'up to 2.16x'; Conclusion says 'up to 1.86x' -> N/A. internal inconsistency in the paper: 1.86x is the continual-pretraining figure

**Transfers to our regime:** no. Assumptions that break: (1) the speedup is draft-and-verify (lossless wrt the LayerSkip model) and needs a model pretrained/finetuned with the recipe, which no available MoE has; (2) the verify pass checks d=4-12 draft tokens at once, which in a sparse MoE touches up to d x 10 distinct experts per layer, so streamed bytes per verified token can rise, not fall (my inference; the paper has no MoE); (3) dense Llama only, GPU (CPU only for a 1.5B model); (4) early-exit alone is the only part that saves bytes and it destroys quality without verification (ROUGE-2 0.012 vs 0.079).

**Quality loss:** Self-speculative output is the same as the LayerSkip model's own greedy output (EM 82.9% at all E in Table 6). Cost is in the model: LayerSkip continual pretraining lowers last-layer accuracy (Llama3 8B MMLU 66.5 -> 60.5; Table 2); for Llama2 7B/13B it is small (Table 1, partly read). Early exit alone is lossy: TOPv2 EM 85.9 -> 79.4 at E=12, 62.9 at E=6.

**Ablation:** Fig 12/appendix: layer-dropout schedule vs constant; Table 7: KV and exit-query cache reuse saves 9-20 ms/token on CPU; self-speculative vs early exit rows in Tables 3-6. No MoE ablation.

**Limitations stated:** Sec 8: requires finetuning or pretraining; hyperparameters pmax, escale, R need tuning; learning rate must be raised for pretraining from scratch.

**Contradicts / supports our findings:** None. Does not conflict with our 1/kept arithmetic; it is a different mechanism and our trace tests cannot test it.

## P-0169 (family #6)

**Title/authors:** ShortGPT: Layers in Large Language Models are More Redundant Than You Expect. Xin Men, Mingyu Xu, Qingyu Zhang, Bingning Wang, Hongyu Lin, Yaojie Lu, Xianpei Han, Weipeng Chen (Baichuan Inc., ISCAS)

**Read:** arXiv 2403.03853v3 (11 Oct 2024), full PDF text, 1091 lines. No venue printed (ACL Findings 2025 is from memory only). (raw_text_read=True)

**Setup:** Training-free layer removal ranked by Block Influence (1 - cosine similarity of layer input/output, computed on PG19). Llama2-7B, Llama2-13B, Baichuan2-7B, Baichuan2-13B (Table 2, ~24-27% parameters removed, same benchmark set and ratios as LaCo), RWKV-7B and Mamba-2.8B (Table 3), Llama2-7B-GPTQ (Table 4), post-training with MLP replacements (Table 6). Removed layers Llama2-13B: 33,31,32,30,29,34,28,35,27,26 (Table 9). Hardware: 2x Xeon 6430, 8x A100 80GB (Table 13); throughput measurement protocol (batch size, seq len, framework) NOT stated. Baselines LLM-Pruner, SliceGPT, LaCo (numbers taken from LaCo paper).

**Headline numbers (what / value / baseline / ratio / where):**
- Llama2-13B MMLU after removing 10 of 40 layers (24.6% params): 52.2 (intro, p2) / 54.69 (Table 2) | baseline: dense 55.00 | ratio: -2.8 points (intro) or -0.31 (table); relative retention averaged over 13 benchmarks 91.64% | p2 intro; Table 2, p5
- Llama2-7B, 9 layers removed (27.1% params), average retention: ShortGPT 86.31% (avg 41.24) | baseline: LaCo 80.39, LLM-Pruner 72.79, SliceGPT 68.73; dense avg 47.78 | ratio: n/a | Table 2, p5
- Llama2-7B generative tasks after 27.1% removal: XSum / C3 / MMLU: 0.67 / 39.62 / 43.96 | baseline: dense 19.40 / 43.56 / 45.39 | ratio: XSum 0.03x of dense | Table 2, p5; Limitation p9
- Llama2-13B XSum / C3 / CoQA after 24.6%: 17.59 / 46.90 / 58.64 | baseline: dense 23.45 / 47.51 / 64.62 | ratio: XSum 0.75x | Table 2, p5
- Llama2-7B-GPTQ, 25% layers removed (24 of 32): throughput: 5045.59 tok/s, perplexity 22.29, MMLU 41.68 | baseline: 4331.23 tok/s, ppl 8.03, MMLU 43.17 | ratio: 1.165x (printed 1.16x); perplexity 2.78x worse | Table 4, p8
- Same, 3.1 / 9.4 / 12.5 / 15.6 / 27.1% removed: 1.02x / 1.06x / 1.08x / 1.10x / 1.19x; ppl 8.37 / 9.44 / 10.24 / 11.42 / 40.78 | baseline: 1.00x | ratio: speedup is below 1/kept (0.75 kept -> 1.33x ideal, measured 1.165x) | Table 4, p8
- Mamba-2.8B / RWKV-7B retention at 25% removed: 90.42% / 73.16% | baseline: 100% | ratio: n/a | Table 3, p8
- Post-training with MLP substitutes, Llama2-7B: avg 43.16 (24.0% removed) | baseline: 41.22 before, dense 47.78 | ratio: +1.94 points | Table 6, p9

**Catalog and first-pass numbers checked:**
- Deleting low-Block-Influence layers without retraining -> YES. Sec 3
- About 25% of layers removed on LLaMA 2-13B (10 of 40) -> YES. intro p2; parameter ratio 24.6% in Table 2
- MMLU 55.0 to 52.2 -> PARTLY. 52.2 only in the introduction; Table 2 of the same paper gives 54.69 (Per. 91.64% overall)
- First-pass: 'Table 4 GPTQ Llama2-7B ~1.16x at 25% removed' -> YES. 1.16x, but Table 4 does not say batch size or hardware
- First-pass: 'generative tasks collapse to near zero at 25% on the 7B models' -> YES. Limitation section p9; XSum 0.67 and 0.04 on Llama2-7B, Baichuan2-7B

**Transfers to our regime:** no. Assumptions that break: dense models (2-13B), permanent removal of the same layers for every input (static), 25% skip gives only 1.165x measured throughput (GPTQ, GPU, setup unstated), 7B models collapse on generation. For a streamed MoE the byte saving would be 1/kept (consistent with our arithmetic) but quality on generative tasks is the open question, and ShortGPT shows it is bad on the small model and tolerable on the 13B. The deeper layers (26-35 of 40) are what it removes, and it has no data for MoE.

**Quality loss:** Table 2: average retention 86.3% (Llama2-7B), 91.6% (Llama2-13B), 85.1% (Baichuan2-7B), 87.4% (Baichuan2-13B) at ~25% removal; MMLU change small (-1.4 on Llama2-7B, -0.3 on 13B), generative tasks large (XSum 19.4 -> 0.67 on 7B). Table 4 shows perplexity rising 8.03 -> 22.29 at 25%.

**Ablation:** Fig 4/5: different importance metrics (BI vs perplexity vs others) on Llama2 and Baichuan2; Table 1: removing FFN vs attention in the last layer (FFN crucial). Table 5: order of pruning vs 4-bit quantization. No separation of speedup contribution.

**Limitations stated:** Sec 5: layer removal hurts generative tasks more than multiple-choice, 7B much more than 13B; cause unexplained; post-training suggested.

**Contradicts / supports our findings:** None. Intro (55.0 -> 52.2) and Table 2 (54.69) disagree with each other. Supports our 'quality unmeasured and decisive' statement; measured speedup (1.165x at 25%) is lower than our 1/kept (1.33x), consistent with a non-bandwidth-bound GPU.

## P-0170 (family #6)

**Title/authors:** The Diminishing Returns of Early-Exit Decoding in Modern LLMs. Rui Wei, Rui Du, Hanfei Yu, Devesh Tiwari, Jian Li, Zhaozhuo Xu, Hao Wang (Stevens Institute of Technology, Northeastern, Stony Brook)

**Read:** arXiv 2603.23701v1 (24 Mar 2026), full PDF text, 905 lines (primary source; catalog URL was the alphaXiv overview). (raw_text_read=True)

**Setup:** Analysis, not a method. Layer-to-final cosine similarity of hidden states, output logits and top-10 sets for exit layers; Early-Exit Adaptability Score EAS = mean over exit layers of S^alpha * skip_ratio^(1-alpha) (linear map of cosine to [0,1]); oracle early exit with logit-similarity threshold chosen to keep accuracy loss within 5%. Models: Llama2-7B, Llama3-8B and more, Llama4-Scout-17B-16E, Qwen2-7B, Qwen3-8B, Qwen3-30B-A3B, GPT-OSS-20B, Mamba1/2, Falcon-H1, Gemma3, Pythia (checkpoints). Datasets: GPQA, GSM8K, HumanEval, MMLU, 100 prompts each, OpenCompass, temperature 0, 1024 max tokens. Hardware: not reported. No latency measured.

**Headline numbers (what / value / baseline / ratio / where):**
- Oracle early exit, 5% accuracy-loss budget, MMLU: Llama2-7B 0.41->0.38 skip 4.69%; Llama3-8B 0.54->0.52 skip 0.18%; Llama4-Scout 0.76->0.73 skip 0.32%; Qwen2-7B 0.24 skip 0%; Qwen3-8B 0.41->0.39 skip 0.10%; OSS-20B 0.74->0.69 skip 7.26% | baseline: full-depth accuracy | ratio: skip fraction of layers (percent) as listed | Table 1, p6
- Oracle early exit, GSM8K: Llama2-7B 0.21->0.17 skip 2.13%; Llama3-8B 0.37->0.33 skip 0.04%; Llama4-Scout 0.85->0.82 skip 0.18%; Qwen3-8B 0.54->0.51 skip 0.62%; OSS-20B 0.61->0.57 skip 4.48% | baseline: full-depth accuracy | ratio: same | Table 1, p6
- EAS (the metric) per model: Llama2-7B 0.52, Llama3-8B 0.46, Llama4-Scout 0.47, Qwen2-7B 0.36, Qwen3-8B 0.51, GPT-OSS-20B 0.59, Mamba2-7B 0.48, LayerSkip-llama3-8B 0.54, EE-LLM-7B 0.56 | baseline: Llama2-7B 0.52 | ratio: relative to Llama2-7B, e.g. Llama3-8B 0.88 | Table 3, p12; Fig 2
- Architecture comparison at comparable scale (the catalog's MoE claim): dense Qwen3-8B EAS 0.51; MoE Qwen3-30B-A3B EAS 0.49; SSM Mamba-Codestral-7B 0.48 | baseline: dense 0.51 | ratio: MoE/dense = 0.96 (difference 0.02) | Fig 7, p7 (text Sec 5.2)

**Catalog and first-pass numbers checked:**
- Qwen3-MoE defers final decisions to deep layers, giving lower early-exit potential -> YES IN DIRECTION, TINY IN SIZE. Sec 5.2: 'MoE models ... defer final decision making to deeper layers'; EAS 0.49 vs 0.51 dense (Fig 7). Abstract: dense offers greater potential than MoE and SSM
- First-pass: 'no numeric EAS for Qwen3-30B-A3B in text' -> WRONG. Fig 7(b) prints MoE (Qwen3-30B-A3B) EAS = 0.49, dense Qwen3-8B 0.51, SSM 0.48
- First-pass: 'GPT-OSS-20B EAS 0.59 highest in Table 3; Llama4-Scout 0.47' -> YES. Table 3 (appendix, p12); Llama4-Scout is the MoE row in Table 1
- First-pass: 'oracle skip only ~4.5-7.3% on GPT-OSS-20B' -> YES. 4.48% (GSM8K) and 7.26% (MMLU)
- Catalog: 'Early-exit analysis for MoE models' -> WRONG FRAMING. The paper is not MoE-specific; MoE is one of three architecture comparisons
- Catalog flag 'secondary source, numbers unverified' -> NOW VERIFIED. primary paper read

**Transfers to our regime:** partly. Evidence that layer-wise early exit potential is small in modern models (oracle skips 0.04-7% of layers within 5% loss) supports our caution about the 'run 16 of 48 layers' idea, including for MoE (Llama4-Scout 0.18-0.32%, OSS-20B 4.5-7.3%). Breaks: early exit (stop at layer l, drop the tail) is not the same as per-sequence or per-token skipping of arbitrary middle layers; no MoE-specific bytes analysis; hidden-state cosine to the final layer, not task quality; 100 prompts per task; hardware not reported.

**Quality loss:** Oracle (cheating) exit: MMLU -0.02 to -0.05, GSM8K -0.03 to -0.04 accuracy at the stated tiny skip fractions; thresholds chosen to keep the loss at most 5%. Practical confidence-based exits are worse (paper text: 'fails to achieve a balance').

**Ablation:** Fig 5: similarity signal (hidden/logit/top-K) vs threshold on Qwen3-8B GSM8K (logit similarity most predictive); Sec 5: scale, architecture, training stage (Pythia checkpoints), base vs instruct (Fig 9), dataset. Nothing isolating expert-level effects.

**Limitations stated:** Limitations section: only methods with little/no training considered (fine-tuned exits or auxiliary heads could do better); training recipes of open models unknown.

**Contradicts / supports our findings:** None numerically. It gives a second weak signal that MoE models are not more skippable than dense; our adjacent-layer expert Jaccard (chance level) is a different quantity and neither contradicts nor supports this.

## P-0172 (family #7)

**Title/authors:** MoE-Infinity: Efficient MoE Inference on Personal Machines with Sparsity-Aware Expert Cache. Leyang Xue, Yao Fu, Zhan Lu, Chuanhao Sun, Luo Mai, Mahesh Marina (University of Edinburgh)

**Read:** arXiv 2401.14361v3 (12 Mar 2025), full PDF text, 772 lines. No venue printed. (raw_text_read=True)

**Setup:** One NVIDIA RTX A5000 24GB GPU, PCIe 4.0 (Table 1 caption says 24 GB/s, hardware paragraph says 32 GB/s), full model in host RAM: 64 GB Switch, 256 GB NLLB, 32 GB DeepSeek, 128 GB Mixtral, 1 TB Arctic. Models: DeepSeek-V2-Lite ('64x2.4B'), Switch-128x0.2B, NLLB-128x0.4B, Arctic-128x4B, Mixtral-8x7B. 290 tasks from BIGBench, FLAN, MMLU; batch size 1; prompt 512, decode 32; TPOT averaged. Baselines: DeepSpeed-Inference, llama.cpp/Ollama (computes offloaded parameters on CPU), Mixtral-Offloading, BrainStorm (their own re-implementation, not open source), vLLM. Algorithm: per-request Expert Activation Matrix (L x E token counts); a collection (EAMC, capacity 120 enough) of past request EAMs; on each layer the current partial EAM is matched by cosine distance to the closest stored EAMs to predict future expert activation; used for prefetch priority and for eviction (layer-discounted probability). Not a static frequency pin. Weights are not quantized; no NVMe/SSD.

**Headline numbers (what / value / baseline / ratio / where):**
- TPOT DeepSeek-V2-Lite, average: MoE-Infinity 155 ms (tail 169 ms; GPU idle 51 ms) | baseline: vLLM 485 ms; DeepSpeed 737; Mixtral-Offloading 1250; Ollama 2590; BrainStorm 934 (text p2) | ratio: 3.13x (vLLM), 4.75x (DeepSpeed), 8.06x (Mixtral-Offloading), 16.7x (Ollama), 6.03x (BrainStorm) | Table 1, p1; text p2
- Mixtral-8x7B decode latency (worst case in paper): 836 ms | baseline: BrainStorm / Mixtral-Offloading comparable | ratio: '1.4x improvement' (text; bar chart only, no table) | Fig 7 and text, p7
- Switch / NLLB / DeepSeek latency vs fully-on-GPU: 155 / 531 / 155 ms | baseline: 'comparable to running fully in GPU' | ratio: n/a | text p7
- Long context DeepSeek-V2-Lite 4K..128K: total compute 50 -> 160 ms; on-demand fetch adds 137 ms at 128K (cache buffer shrinks 3 -> 1 GB) | baseline: vLLM/Mixtral-Offloading worse | ratio: n/a | Fig 8 and text, p8
- Experts reused within one request: <5% of experts repeatedly activated per request for ~100-expert models; ~25% for Mixtral | baseline: n/a | ratio: n/a | Sec 4.1, p3
- Reuse skew across requests: skew disappears over 1000 requests (uniform) so global frequency (BrainStorm-style) fails | baseline: within-request skew | ratio: n/a | Sec 4.2, Fig 2, p3-4
- Recovery after workload shift: ~50 requests within a dataset (5% worst case), ~30 requests across datasets | baseline: n/a | ratio: n/a | Table 3, p8
- Abstract vs conclusion range: abstract 3.1-16.7x; conclusion 2.7-13.7x | baseline: n/a | ratio: paper inconsistent with itself | abstract p1; Sec 6, p8

**Catalog and first-pass numbers checked:**
- 3.1-16.7x lower per-token latency -> YES (ABSTRACT), CONCLUSION SAYS 2.7-13.7X. 3.1x = vs vLLM, 16.7x = vs Ollama (Table 1, DeepSeek-V2-Lite, TPOT average)
- compared with vLLM, Ollama, DeepSpeed and BrainStorm -> YES. abstract list; Mixtral-Offloading also a baseline (8.06x)
- Request-level expert activation counts guide cache replacement and prefetch -> YES, BUT INCOMPLETE. it is matched activation matrices (cosine) against stored past requests, not raw counts; and it is not a static pin
- First-pass: Mixtral worst case ~836 ms only 1.4x -> YES. text p7
- First-pass: 'weak baselines' -> PARTLY. baselines include purpose-built Mixtral-Offloading; but 16.7x is vs Ollama/llama.cpp, whose offloaded compute runs on CPU

**Transfers to our regime:** no. Assumptions that break: GPU 24GB with the whole model in host RAM over PCIe 4.0 (not NVMe, not CPU compute); expert cache measured in GB of VRAM; models 64-128 experts with top-1..top-6 (DeepSeek-V2-Lite) so small per-request working sets; weights fp16; 290 tasks batch 1, prompt 512 decode 32 only (short); the cache holds a share of experts decided by VRAM (Mixtral 25%/request). The cross-request reuse finding (Sec 4.2) is the transferable part and agrees with ours: a hot set learned on other prompts is useless.

**Quality loss:** None (lossless caching/prefetch).

**Ablation:** None isolating cache vs prefetch vs layer-aware eviction (Fig 6 is an illustrative example only). Only EAMC capacity (Fig 9) and workload-shift (Table 3) micro-benchmarks.

**Limitations stated:** No limitations section. Text notes Mixtral (few large experts) is its worst case and long context reduces cache buffer.

**Contradicts / supports our findings:** Supports: within a request experts are reused, across requests uniform; our other-prompt pin lost by 0.15-0.29 hit rate. Does not test our within-request static pin vs LRU, since LRU baselines are in vLLM/Ollama with dense-layer-order prefetch; they never compare against a warmed LRU alone, so our 'pin ~ LRU' result neither contradicts nor is covered. Their 'Ollama' CPU-compute baseline is the CPU llama.cpp regime closest to ours and is the slowest (2590 ms) in their setup.

## P-0174 (family #8)

**Title/authors:** KBVQ-MoE: KLT-guided SVD with Bias-Corrected Vector Quantization for MoE Large Language Models. Zukang Xu, Zhixiong Zhao, Xing Hu, Zhixuan Chen, Dawei Yang (Houmo AI, NTU)

**Read:** arXiv 2602.11184v2 (24 Feb 2026), header 'Published as a conference paper at ICLR 2026', full PDF text, 1380 lines. (raw_text_read=True)

**Setup:** PTQ with vector quantization: IDRE (KLT-guided SVD extracting a shared low-rank component across experts kept at higher precision, rank = 1/128 of full) then VQ of the residual (vector length 4, k-means++ 100 iterations), BCOS (channel-wise affine bias correction). Models: Qwen1.5-MoE-A2.7B, Qwen3-30B-A3B, Mixtral-8x7B, DeepSeek-V2-Lite (and DeepSeekMoE-16B in Table 17). Calibration 256 RedPajama samples x 4096. Evaluation lm-eval-harness 7 zero-shot tasks + WikiText2 PPL. Hardware: 'NVIDIA RTX A6000' (Sec 5; appendix A.10 says 'RTX A100', inconsistent). Effective bits 2.08 for '2-bit' (Table 8: Qwen1.5 27.9 GB -> 4.3 GB, Qwen3-30B 59.1 -> 8.7 GB). Baselines RTN, GPTQ (2.25 bits in Table 8), MoEQuant, plain VQ (also 2.08 bits).

**Headline numbers (what / value / baseline / ratio / where):**
- Qwen1.5-MoE-A2.7B, 2-bit: Wiki2 PPL / 7-task average: KBVQ-MoE 9.61 / 62.78 | baseline: FP16 7.22 / 68.07; GPTQ 12.69 / 49.07; plain VQ 26.95 / 47.84; MoEQuant 583542 / 34.64; RTN 638509 / 25.64 | ratio: -5.29 points vs FP16; +19.0 vs GPTQ and +20.2 vs plain VQ in points | Table 1 p8; Table 10 p19
- Same model, 3-bit: KBVQ-MoE 7.74 / 67.99 | baseline: FP16 7.22 / 68.07; GPTQ 7.58 / 66.36; MoEQuant 7.87 / 66.79; plain VQ 11.47 / 55.94 | ratio: -0.08 vs FP16; +1.6 vs GPTQ; +12.1 vs plain VQ | Table 1 p8; Table 10 p19
- Qwen3-30B-A3B, 2-bit / 3-bit average: KBVQ-MoE 11.87 / 63.37 (2-bit); 9.26 / 69.09 (3-bit) | baseline: FP16 8.70 / 70.24; 2-bit GPTQ 438.4 / 33.06, plain VQ 115.3 / 30.61; 3-bit GPTQ 11.32 / 63.51, plain VQ 18.72 / 52.06 | ratio: 2-bit -6.87 points; 3-bit -1.15 points | Table 1 p8; Table 11
- Mixtral-8x7B 2-bit / 3-bit: KBVQ 5.39 / 75.69; 4.07 / 78.35 | baseline: FP16 3.88 / 78.57; 2-bit GPTQ 5.69 / 67.56, VQ 5.99 / 59.22 | ratio: 2-bit -2.9 points | Table 1 p8
- Ablation Qwen3-30B-A3B 3-bit: neither 18.72 PPL; IDRE only 11.67; BCOS only 14.32; both 9.26 | baseline: plain VQ 18.72 (ARC-E 57.83) | ratio: IDRE is the main gain | Table 2 p8
- Rank ablation Qwen3-30B-A3B 2-bit: k/n 0: 2.01 bits PPL 15.30; 1/128: 2.08 bits 11.87; 1/64: 2.11 bits 11.30; 1/32: 2.20 bits 11.01 | baseline: n/a | ratio: n/a | Table 4 p9
- Decoder speed, 1k input tokens: Qwen1.5-MoE 35.24 tok/s; Qwen3-30B-A3B 17.37 tok/s | baseline: BF16 22.31; 10.85 | ratio: 1.58x; 1.60x | Table 6 p9
- Challenging benchmarks 2-bit Qwen3-30B-A3B: MMLU 68.19, GSM8K 78.92, HumanEval 66.89 | baseline: FP16 79.5, 85.44, 71.5; plain VQ 60.94, 64.83, 61.2 | ratio: -11.3, -6.5, -4.6 points | Table 17 (appendix)

**Catalog and first-pass numbers checked:**
- VQ on MoE experts needs shared-component extraction and bias correction -> YES. IDRE + BCOS, ablation Table 2
- near-FP16 at 3 bits on Qwen1.5-MoE-A2.7B -> YES. 67.99 vs 68.07
- builds on AQLM and VPTQ -> PARTLY. cites AQLM (residual quantization), VPTQ and GPTVQ; Table 5 plugs IDRE/BCOS into GPTVQ and VPTQ, it does not build on AQLM code
- First-pass: '2-bit KBVQ 9.61/62.78 vs plain VQ 11.47/55.94 and GPTQ 2-bit 7.58/66.36' -> WRONG. 11.47/55.94 and 7.58/66.36 are the 3-bit rows (Table 1, Table 10). At 2-bit plain VQ is 26.95 / 47.84 and GPTQ 12.69 / 49.07
- First-pass caveat: 'At 2-bit its own table shows GPTQ and MoEQuant beating it' -> WRONG. KBVQ-MoE beats every baseline at 2-bit and 3-bit on all four models (GPTQ 49.07, MoEQuant 34.64 at 2-bit on Qwen1.5). Only at 4 bits is it merely competitive (Table 9)
- First-pass: 2.08 effective bits; 1.58x speed -> YES. Table 8; Table 6 (hardware A6000 per Sec 5, kernels unnamed, BF16 baseline)
- First-pass: 'uses 4-dim vectors' -> YES. vector length 4
- First-pass: 'quality at 1.3 bpw not shown' -> YES. Limitation 3: 1-bit and 1.5-bit untested

**Transfers to our regime:** partly. Evidence (not speed) transfers: it quantifies how much plain vector quantization loses at 2.08 bits on the very MoE we trace (Qwen3-30B-A3B: plain 4-dim VQ 30.61 average accuracy vs FP16 70.24, PPL 115). Our probe at 1.0 bpw (8-dim, 256 words, no calibration) is a harsher regime on the same codebook size, and the paper gives no number below 2.01 bits. The 1.58x speed number does not transfer: GPU A6000, BF16 baseline (note the BF16 Qwen3-30B-A3B at 59.1 GB exceeds a 48 GB A6000, so that baseline must be offloaded or split, my inference), custom unnamed kernels, no CPU. Also needs calibration data, shared full-precision low-rank component (+0.07 bits), per-model offline cost.

**Quality loss:** Qwen1.5-MoE: -5.3 points at 2.08 bits, -0.08 at 3 bits; Qwen3-30B-A3B -6.9 and -1.2; Mixtral -2.9 at 2 bits; benchmarks Table 17 show -4.6 to -11.3 points on MMLU/GSM8K/HumanEval at 2 bits (Qwen3-30B-A3B). Plain VQ at 2.08 bits loses 20 points (Qwen1.5) and 39.6 points (Qwen3-30B-A3B).

**Ablation:** Tables 2-5, 14, 15: IDRE vs BCOS vs both, KLT vs plain SVD, rank, plug-in to GPTVQ/VPTQ. Nothing on codebook size or vector length.

**Limitations stated:** A.12: rank chosen empirically; decoder-only only; no 1-bit/1.5-bit regime tested. (Text confirms hybrid/extreme regimes would need stronger codebooks.)

**Contradicts / supports our findings:** Strongly consistent: plain VQ is not usable at ~2 bits on Qwen3-30B-A3B, so plain 8-dim at 1.0-1.3 bpw is worse still (our probe error 0.58). First-pass framing 'GPTQ beats KBVQ at 2 bits' must be dropped; it does not weaken our probe.

## P-0175 (family #8)

**Title/authors:** QMoE: Practical Sub-1-Bit Compression of Trillion-Parameter Models. Elias Frantar, Dan Alistarh (IST Austria)

**Read:** arXiv 2310.16795v1 (25 Oct 2023), full PDF text, 898 lines. Venue not printed (MLSys 2024 from memory). (raw_text_read=True)

**Setup:** Data-dependent (GPTQ-style) quantization of SwitchTransformer base128, large128, c2048 (1.6T params, 2048 experts, top-1, T5 encoder-decoder, MLM pretraining) to 2-bit and ternary, then a custom dictionary/entropy code (pairs of ternary values, fixed 16-bit-ish dictionary in GPU L2) with fused GPU decode+matvec kernels. Compression: one A6000 48GB, under a day (Table 8: 20.8 h for c2048 at 20K/320K calibration samples), few hundred GB CPU RAM. Inference: 4x A6000 or 8x RTX 3090, HuggingFace, offloading to CPU memory for sparse layers in compression stage. Metrics: C4 validation loss (128 samples), plus Arxiv/Github/StackExchange/Wikipedia loss; no zero-shot accuracy. Baselines: BF16, RTN (identity Hessian).

**Headline numbers (what / value / baseline / ratio / where):**
- C4 validation loss c2048: QMoE 2-bit 1.20, ternary 1.26 (1.0x calibration) | baseline: BF16 1.18; RTN 2-bit 1.33, ternary 2.15 | ratio: ternary +6.8% over BF16 (text says 6.7%); 2-bit +1.7% | Table 5, p9
- C4 loss base128 / large128 ternary: 1.99 / 1.69 | baseline: BF16 1.73 / 1.55 | ratio: +15.0% / +9.0% | Table 5, p9
- Natural sparsity of ternary weights: c2048 88.6% zeros (2-bit 76.5%); base128 85.7% | baseline: n/a | ratio: n/a | Table 4, p5
- Compression c2048: 3142 GB -> 158.6 GB (full model 19.81x, MoE-only 20.07x) = 0.807 bits/param | baseline: bf16 | ratio: 19.81x | Table 7, p9
- Other-domain loss c2048: Arxiv 1.42, Github 1.13, StackExch 1.22, Wiki 1.32 (ternary) | baseline: BF16 1.31, 0.99, 1.15, 1.20 | ratio: +8%, +14%, +6%, +10% | Table 6, p9
- End-to-end overhead (128 generated tokens): <5% slower | baseline: BF16 estimate obtained by pointing all experts of a layer at the same weight memory ('highly optimistic', lower bound) | ratio: n/a | Fig 5 right and text, p10
- Per-layer fused kernel: up to 35% faster than BF16 on some shapes (smallest matrix <0.02 ms) | baseline: PyTorch cuBLAS bf16 | ratio: <=1.35x | Fig 5 left, p10
- Compression runtime: 13.3 h / 16.0 h / 20.8 h for 5K-80K / 10K-160K / 20K-320K samples | baseline: n/a | ratio: n/a | Table 8, p10

**Catalog and first-pass numbers checked:**
- 0.8 bits per parameter -> YES. 0.807 (Table 7)
- 1.6T SwitchTransformer to under 160 GB (~20x) -> YES. 158.6 GB, 19.81x
- no retraining -> YES. post-training, data-dependent quantization with calibration data
- dictionary coding on ternary weights with GPU decode kernels -> YES. Sec 4
- First-pass: '<5% overhead vs uncompressed (own estimate)' -> YES. baseline is an estimated lower bound, not a run
- First-pass: 'mixtral-offloading authors report sub-1-bit QMoE gave too much perplexity loss on Mixtral' -> NOT CHECKABLE HERE. that claim is about another paper; not in QMoE text (from first pass only)

**Transfers to our regime:** no. Assumptions that break: 0.8 bits exists only because 88.6% of ternary-rounded weights are zero (natural sparsity grows with expert count: 85.7% at 128 experts, 88.6% at 2048), which is a property of Switch c2048's tiny, redundant experts; ternary loss already grows from +6.8% (2048 experts) to +15% (128 experts); the T5 encoder-decoder MLM model is not a decoder LLM with 512 experts top-10; custom GPU kernel (GPU L2-resident dictionary); no CPU decoder; no downstream accuracy. A 512-expert top-10 decoder sits between the 128 and 2048 expert Switch points, nearer the lossy end.

**Quality loss:** C4 validation loss only: +6.8% (c2048 ternary), +1.7% (2-bit), +9% (large128), +15% (base128); RTN ternary +82% on c2048. No benchmark accuracy.

**Ablation:** Tables 2-3: special-token masking (base128 ternary 2.16 -> 1.99), activation reordering and sequential execution (heuristics, ineffective). Calibration sample count (Table 5 0.5x/1x/2x). No ablation of dictionary vs plain ternary packing on loss.

**Limitations stated:** Sec 7: only SwitchTransformers evaluated, bespoke frameworks hinder other MoEs; compressed-model finetuning unexplored.

**Contradicts / supports our findings:** None. Consistent with our 'not comparable, not tested' status; adds that gains scale with expert count in the paper's own data (ternary loss grows as experts shrink), a useful prior for 512 experts.

## P-0176 (family #8)

**Title/authors:** Mixture of Quantized Experts (MoQE): Complementary Effect of Low-bit Quantization and Robustness. Young Jin Kim, Raffy Fahim, Hany Hassan Awadalla (Microsoft)

**Read:** arXiv 2310.02410v1 (3 Oct 2023), full PDF text, 623 lines. Footer: 'Preprint. Under review.' (raw_text_read=True)

**Setup:** Weight-only linear (channel-wise) quantization of expert FFN weights only (rest fp16). Multilingual machine-translation Transformer, 5.3B-parameter MoE with 32 experts, top-1 gating, MoE every other layer, vs FLOPs-equivalent dense model; 200,000 updates on internal 6-language data (Table 2, 6, 7) and WMT-10 public (Table 5). Metric BLEU (sacrebleu). Hardware: single NVIDIA A100, CUDA 11.6. Throughput batch size 24 sentences (Table 1). No kernel below 4 bit. 2-bit needs QAT.

**Headline numbers (what / value / baseline / ratio / where):**
- Average BLEU and size/throughput by precision (Table 2): dense fp16 45.06; MoE fp16 46.35; int8 46.34 (1.16x thr, 0.54x size); int4 46.18 (1.24x, 0.32x); int3 46.01 (not optimized, 0.26x); int2 QAT 45.90 (not optimized, 0.20x) | baseline: MoE fp16 46.35 (1.00x, 1.00x) | ratio: int2 QAT is -0.45 BLEU (-0.97%) vs MoE fp16 and +0.84 BLEU vs dense fp16; size 0.20x = 79.6% reduction | Table 2, p4
- MoE vs dense inference cost: MoE 5.37 sentences/s, 9.91 GB; dense 14.02 sentences/s, 1.18 GB | baseline: dense | ratio: 0.38x throughput, 8.38x size | Table 1, p1
- Post-training (no QAT) BLEU change on WMT-10, MoE experts, en-xx / xx-en averages, PERCENT: 8-bit +0.05/0.00; 4-bit -0.36/+0.01; 3-bit -1.21/-1.01; 2-bit -12.34/-11.54 (percent) | baseline: fp16 MoE 22.55 / 26.71 BLEU | ratio: 2-bit = about -2.8 / -3.1 BLEU by my calculation; dense 2-bit -91.21% / -81.33% (collapse) | Table 5, p10
- Post-training on internal 5.3B data, PERCENT: 2-bit MoE -3.14% (xx-en) / -5.28% (en-xx), dense -94.8% / -96.1%; 3-bit MoE -0.38% / -1.10% | baseline: fp16 MoE 48.01 / 44.69 BLEU | ratio: n/a | Table 6, p10; Table 7 -4.60% / -6.25%
- Text claim for Fig 3: MoE within -0.3 down to 3-bit and -1.82 for 2-bit; dense loses 42.96 BLEU at 2-bit | baseline: n/a | ratio: not reproducible from Table 5 (which gives about -12% = -2.8 BLEU at 2-bit) | Sec 3.2, p4
- Weight skewness evidence: expert fc2 mean skew -0.63/+0.48 vs non-expert fc2 -1.84 (and -10.7 on one layer) | baseline: n/a | ratio: n/a | Table 3, p8

**Catalog and first-pass numbers checked:**
- 2-bit experts only -> YES, WITH QAT. 'int2 QAT' row; 2-bit without QAT is lossy (see below)
- 79.6% size cut -> YES. abstract/intro; 0.20x in Table 2, requires QAT
- scalar -> YES. linear channel-wise scalar quantization (log-scale tried, Fig 4)
- experts tolerate low bits better than dense FFN -> YES. Fig 3 and Tables 5-7: dense 2-bit collapses (-81..-97%), MoE -3..-12%
- First-pass: 'Table 5 public WMT 2-bit drops ~11-12 BLEU' -> WRONG (UNIT). Table 5 values are percent differences: -12.34% and -11.54% (about 2.8-3.1 BLEU), not 11-12 BLEU
- First-pass: 'post-training 2-bit drops 1.82 BLEU in one setting' -> PARTLY. 1.82 is the text claim for Fig 3 (BLEU); tables give percent drops of 3.1-12.3%
- First-pass: int4 46.18 / 1.24x / 0.32x -> YES. Table 2
- First-pass: 'headline int2 uses QAT; catalog omits it' -> YES. abstract and Sec 3.1

**Transfers to our regime:** no. Assumptions that break: machine-translation encoder-decoder with 32 small experts, top-1; 5.3B params; not an LLM; BLEU not a language-model quality measure; headline 2-bit needs QAT (retraining); no optimized kernel below 4 bit, speedup (1.24x) is 4-bit on A100 at batch 24. Qualitative message (experts have fewer outliers, so post-training 3-bit is nearly free and 2-bit costs 3-12% BLEU where dense collapses) is the only transferable part; it is a different architecture from a 512-expert top-10 decoder.

**Quality loss:** Post-training: 3-bit -0.4..-1.2% BLEU; 2-bit -3..-12% BLEU (depending on dataset). With QAT 2-bit: -0.97% vs fp16 MoE (45.90 vs 46.35).

**Ablation:** Fig 2 component sensitivity (experts vs dense FFN vs attention), Fig 4 linear vs log quantization, Fig 5 channel-wise vs matrix-wise scaling, Fig 6 which dense layers; no cache or layout ablation.

**Limitations stated:** Sec 4: no optimized implementation below 4 bits; 2-bit requires QAT; hybrid precision across layers not explored.

**Contradicts / supports our findings:** None. Our closed-form byte-lever (2.0 bpw: 1.75x serial / 1.09x overlap at 8 GB) is not tested by this paper. Correction to the first pass: the 'quantization hurts a lot post-training' impression came from a percent-vs-BLEU unit error; the real PTQ 2-bit damage is 3-12% BLEU relative.

## Family #6 layer skipping / early exit: prior art and novelty

**Further prior art:**
- FlexiDepth (Luo, Wang, Yan; COLM 2025; arXiv 2503.23798v3) READ RAW: router+adapter on frozen Llama-3-8B-Instruct skips 8 of 32 layers while Table 1 average retention 100.7% (MMLU 0.616 vs 0.673, GSM8K 0.662 vs 0.679, HumanEval 0.341 vs 0.299); fixed-k baselines at 8 skipped layers: ShortGPT 32.0%, LayerSkip 43.9%, LaCo 65.3%, MindSkip 60.2% retention (Table 1). Per-token routing, layers skipped vary by task. States in the abstract and Limitations that it does NOT achieve wall-clock speedup on GPU (irregular skipping, KV for skipped tokens, I/O). Closest published evidence that token-level skipping gives FLOP not time savings; dense only.
- Router-Tuning (He et al.; EMNLP 2025; arXiv 2410.13184v6) READ RAW key tables: router-only tuning for dynamic skipping; Table 2 Llama-3-8B: skipping Attention 1.18-1.21x with ARC-C 56.4-56.6 vs 58.1, but skipping Block 1.24-1.26x drops ARC-C to 44.2-44.5 and HellaSwag 82.1 -> 62.6-63.1, MLP skipping only 1.05x with big drop. Table 7 (MoE): token-level router tuning at 25% skipping on OLMoE gives 1.12x speedup, average 64.0 -> 61.4; DeepSeek-MoE 1.10x, 63.7 -> 62.4; static Expert Drop 1.13x, OLMoE 64.0 -> 50.2. So the only MoE data point on layer/expert skipping is ~1.1x at about 4-6% relative quality loss, A6000/A100 GPU.
- LayerSkip / ShortGPT / Dr.LLM / early-exit analysis: the four rows above.
- Not All Experts are Equal (Lu et al., ACL 2024, arXiv 2402.14800): expert-level skipping in Mixtral; found by search, opened only as a result listing (UNREAD in this pass).
- LayerRoute (arXiv 2609.13682): mean decode speedup ~1.04x (1.02-1.06 across seeds) per search snippet (UNREAD, snippet only).
- MoDES (CVPR 2026, arXiv 2511.15690): dynamic expert skipping, 1.26x decode on Qwen3-VL-MoE per snippet (UNREAD). ASET (ACL Findings 2026): adaptive top-k, 10-78% throughput on fine-grained MoEs per snippet (UNREAD).
- From memory (not opened): Mixture-of-Depths (Raposo et al. 2024), CALM (Schuster et al. 2022), LayerDrop (Fan et al. 2020), Gromov et al. 2024 'Unreasonable ineffectiveness of the deeper layers', SkipDecode, FFN-SkipLLM.

**What is new in our variant:** Not found in any paper read: skipping whole transformer blocks specifically to avoid streaming that layer's experts from disk in a bandwidth-bound MoE, and no paper measures layer skipping on a 48-layer 512-expert top-10 model. The closest papers are Router-Tuning (MoE token-level skipping, 1.10-1.12x on GPU) and FlexiDepth (dense, no wall-clock gain). What would be new is only the regime (bytes dominate, so the speedup is 1/kept instead of FLOP-limited), which our arithmetic already gives; the open part is quality at 25-67% skip, which the best published adaptive method (FlexiDepth) holds only at 25% (8 of 32 layers) on a dense 8B after training a router plus adapter.

## Family #7 session-pinned RAM set: prior art and novelty

**Further prior art:**
- FlashMoE (Kim, Lee, Yoo, Kim; arXiv 2601.17063v2, 5 Oct 2026) READ key sections: SSD-offloaded experts on desktop (Ryzen 5 9600X, RTX 5070 Ti 16GB, 4x16 GB DDR5 with ~1 GB left free, PCIe 5.0 NVMe 7.4 GB/s); per-layer ML cache FFN (113 KB) trained on Belady labels from TriviaQA. LRU hit rate 73% vs Belady 86% on its example; vs best heuristic (LRU) token throughput +22% OLMoE-1B-7B and +7% Qwen3-30B-A3B (Fig 7, Sec 4.4); hit-rate gain 21% over LRU and 51% over LFU for OLMoE. Pin-from-training-popularity baselines (Fiddler/DAOP) are slower. This supports: a smart policy beats LRU by only ~7% on Qwen3-30B-A3B.
- DALI (Zhu et al.; arXiv 2602.03495v1, Feb 2026) abstract and intro read: dynamic CPU/GPU expert assignment + workload-aware cache; claims decode speedups 3.97x, 2.16x, 1.48x, 1.32x vs llama.cpp, KTransformers, MoE-Lightning, HybriMoE; GPU-centric; no NVMe (host RAM). Weaker threat to novelty than FlashMoE.
- HOBBIT (arXiv 2411.01433), SpecMD, 'Fewer Expert Transfers Without Changing Routing', 'Paging the Experts', SeqMoE: as listed in the first pass; not re-read here (first-pass level only).
- SSD-offloading energy paper (arXiv 2508.06978), ProMoE (arXiv 2410.22134, llama.cpp-integrated proactive caching), llama.cpp discussion #23324 (RFC for on-demand expert paging) and Flash-MoE Apple-silicon blog: seen as search results only (UNREAD).
- Mixtral-Offloading (Eliseev, Mazur) LRU + speculative prefetch: first-pass level.

**What is new in our variant:** MoE-Infinity (read in full) already claims the core observation that expert reuse is a within-request property that vanishes across requests, and builds a per-request tracked cache; FlashMoE covers the SSD regime with a learned policy and shows only +7% over LRU on Qwen3-30B-A3B. What may be new in our variant: using the prefill-phase routing to pin a RAM set sized to 8-32 GB on CPU-only NVMe streaming for a 512-expert top-10 model; no paper read tests that, and our trace result (pin ~ warmed LRU) says it adds little. Closest papers: MoE-Infinity and FlashMoE.

## Family #8 8-dim vector quantisation at 1.3 bpw: prior art and novelty

**Further prior art:**
- AQLM (Egiazarian et al., ICML 2024, arXiv 2401.06118) READ key sections: additive multi-codebook quantization with fine-tuning. Table 5: CPU (Intel i9 13900K, 8 cores) matvec 2.75x/3.54x/3.69x faster than FP32 for 2x8-bit codebooks (7B/13B/70B), up to 4.07x with 4x8-bit; Table 14 end-to-end batch 1 on i9: Llama2-7B FP32 3.106 tok/s vs AQLM 2x8 6.961 (2.24x), 70B 0.297 vs 0.966. Quality: 2x8 codebooks at 2.0 bits is worse than the 1x16 variant; Mixtral-8x7B at 1.98 bits average accuracy 67.68 vs 72.33 FP16 (Table 3). CPU lookup-table VQ already exists, at 2 bits and with fine-tuning.
- CCQ / convolutional-code quantization for ERNIE-4.5-300B-A47B (Baidu, arXiv 2507.07145): abstract and introduction read; claims 2.0-2.75 bits with minimal loss, a lookup-free codebook (bit-shift decode) to avoid VQ's lookup cost, 2-bit ERNIE-4.5-300B-A47B at 89 GB on a single node; tables not read in this pass (PARTLY READ).
- VPTQ, GPTVQ, Mixture Compressor (MC-MoE), TriS-MoE: first-pass level; MC-MoE per search is HQQ scalar per-expert mixed 1.5-2.5 bits, not codebook.
- From memory: QuIP#, QTIP, MoEQuant, EAQuant.

**What is new in our variant:** Nothing found that tests plain 8-dim k-means VQ at ~1 bpw on MoE experts; published VQ for MoE starts at 2.0 bits and needs shared components, bias correction or fine-tuning, and its plain-VQ baseline is already bad at 2.08 bits (KBVQ Table 1). Closest papers: KBVQ-MoE (read) and AQLM (read) for CPU kernels. The only non-obvious contribution of our family would be 1.0-1.3 bpw on CPU from NVMe, for which no published quality number exists and our own probe says it fails.
