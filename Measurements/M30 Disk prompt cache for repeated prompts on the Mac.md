---
id: M30
type: measurement
status: done
links: [Pool warm start and adaptive pool size on the Mac]
date: 2026-10-30
---
# Disk prompt cache for repeated prompts on the Mac

llama-completion --prompt-cache FILE, pool+overlap 1500 MB, 9 GB wired ballast, Qwen3-30B Q4_K_M, a 330-word (about 450 token) prompt, 16 generated tokens. No cache: prompt eval 52.0 s, wall 62.9 s. Cache save (first run): 53.1 s, 65.2 s. Cache load on a repeated prompt (--prompt-cache-ro): prompt eval 0 ms, wall 11.1 s (5.7x faster end to end, time to first token about 62 s to under 11 s). Decode speed unchanged (1.57-1.69 tok/s). Cache file 41.2 MB for about 450 tokens. Caveat: the generated text with the cache flag (md5 7158bf6a, identical for save and load) differs from the text without it (md5 41925412, identical across two runs), so the cached path has a small numerical difference; not investigated. Verdict: confirmed as a time-to-first-token win for repeated prompts or shared prefixes (agent loops), no help on a miss and none for decode.
