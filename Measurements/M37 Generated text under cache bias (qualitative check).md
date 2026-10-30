---
id: M37
type: measurement
status: done
links: [Cache-aware routing on a second model geometry (OLMoE, 64 experts)]
date: 2026-10-30
---
# Generated text under cache bias (qualitative check)

Qwen3-30B Q4_K_M, pool 3000 MB, greedy, prompt asking for three short paragraphs on why the sky is blue and sunsets are red, 110 tokens. No bias, bias 1.0 and bias 3.0 produced the same explanation (Rayleigh scattering, longer path at sunset) with small wording differences; all fluent and correct. 6 experts + bias 3.0 followed a different but equally fluent continuation (it restated the task and started reasoning). No degeneration or repetition seen. A small qualitative check only; not a benchmark.
