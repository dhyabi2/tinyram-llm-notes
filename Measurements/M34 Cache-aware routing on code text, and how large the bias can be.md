---
id: M34
type: measurement
status: done
links: [Cache-aware routing in the real runtime (selection bias toward resident experts)]
date: 2026-10-30
---
# Cache-aware routing on code text, and how large the bias can be

Same setup as the documentation-text run, on a code corpus (440 KB of Python source), decode-mode perplexity, 4 chunks of 512 tokens, pool 1500 MB. No bias 4.053 (misses 36,220, hit 91 percent); bias 3.0 4.241 (+4.6 percent, misses 2,843, 92 percent fewer); bias 10.0 4.880 (+20.4 percent, misses 967). On documentation text the cost at 3.0 was +0.03 percent, at 10 not measured. Conclusion: the quality cost of the selection bias depends on the text: negligible on prose documentation, about 5 percent on code at lambda 3, 20 percent at lambda 10. A default of lambda about 1 (documentation +0.07 percent, 72 percent hit rate) is conservative; lambda 3 gives 81 percent hit and 1.85x speed with up to about 5 percent perplexity loss on code. Use together with a bigger cache when quality matters; the best lambda should be chosen per workload with a small perplexity check (a few minutes).
