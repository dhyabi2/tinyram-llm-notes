---
id: F05
type: fix
status: done
links: [L04,T01,T02]
date: 2026-10-28
---
# tokens counted

llama-bench runs a 1-token warmup, so read per token divides by tokens+1; result files record `tokens_counted`.
