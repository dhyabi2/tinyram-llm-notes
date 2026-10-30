---
id: M41
type: measurement
status: done
links: [Best configurations across eight prompt types]
date: 2026-10-30
---
# Task check- 30 short questions under the best configurations

Qwen3-30B Q4_K_M, pool 2500 MB with adaptive size, greedy, Qwen3 non-thinking chat format, 28 generated tokens, 30 questions with unambiguous short answers (arithmetic, capitals, chemistry symbols, Python semantics, simple facts); an answer counts when the expected string appears in the first 120 characters. Baseline (pool, 8 experts) 28/30; 6 experts + bias 1.0 28/30; 6 experts + bias 3.0 28/30; 8 experts + bias 3.0 28/30. An earlier version of the test with a raw prompt and a 14-token window was invalid (the model began 'Okay, so the user is asking' and ran out of tokens; scores 19-21/30 flipped both ways), so only the chat-format run is reported. Verdict: no measurable task-quality loss at this resolution (n=30, one standard error about 6 points), consistent with the small perplexity changes; a larger benchmark would be needed for claims below a few percent.
