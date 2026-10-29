---
id: F16
type: fix
status: done
links: [T06]
date: 2026-10-28
---
# quant map norm pin

Stock llama-quantize quantizes 2-D norm tensors in qwen4_exp and the model then aborts at run time; the map tool pins them to f32.
