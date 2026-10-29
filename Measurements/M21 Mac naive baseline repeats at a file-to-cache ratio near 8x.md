---
id: M21
type: measurement
status: done
links: [Mac naive baseline under a wired ballast]
date: 2026-10-29
---
# Mac naive baseline repeats at a file-to-cache ratio near 8x

Qwen3-30B (17.3 GiB file) plain mmap on the 16 GB Mac, 8 tokens, 3 reps per size, wired ballast, residency checked with mincore after each run. A 9 GB wired ballast leaves only 1.8-2.45 GiB of the file resident (10-14%, a file-to-cache ratio of about 7-9x, close to the real 8.6x target): 0.277 / 0.325 / 0.399 tok/s (median 0.325), 0.76-1.03 GiB paged in per token. Reference runs with no ballast: first run 0.83 tok/s with 43% of the file resident, second run 8.0 tok/s fully warm (0.002 GiB per token). A 10 GB ballast does not wire (wire limit reached, 0% wired), those three runs are invalid (1.27-1.37 tok/s, nothing squeezed) and are excluded. So on the Mac the naive baseline falls from 8.0 tok/s (warm) to about 0.33 tok/s at the 8x ratio, 24x lower, while a cold unsqueezed start is 0.83. Caveats: 8-token runs, spread 0.28-0.40, swap rose by about 1 GB during the 9 GB runs, 30B model not the 125B. This is the first naive baseline at the real ratio; the modelled base path for the 125B at 8 GB is 2.49 tok/s.
