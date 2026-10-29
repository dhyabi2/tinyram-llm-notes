# Trace-driven combinations (measured cache behaviour, projected speed)

3888 runs = 3 recorded Qwen3-30B router traces x 3 cache policies x 4 expert-selection rules x 3 bit widths x 3 speculation settings x 2 layer-skip settings x overlap on/off x 3 RAM sizes. Hit rates are simulated on the real traces; speeds apply the time model of combo_model.py (scenario numbers there stay estimates). Quality: only measured OLMoE perplexity losses are used; `?` means unmeasured.


## 8 GB machine - base path 2.50 tok/s (likely scenario, mean of 3 traces)

Single levers (x base): slru 0.98, pin 0.98, top6 1.24, top4 1.61, aware6 1.38, bits3.5 1.17, bits3 1.29, spec2 0.93, spec4 0.74, skip25 1.47, overlap 1.26

Pairs: 13 better than the product of their parts (>3%), 11 worse (<-3%), 25 about equal.

Best synergies: bits3+spec4 1.27 vs product 0.95; spec4+skip25 1.39 vs product 1.08; top4+spec4 1.52 vs product 1.19; aware6+spec4 1.29 vs product 1.02; top6+spec4 1.16 vs product 0.92; bits3.5+spec4 1.06 vs product 0.86

Worst interference: spec4+overlap 0.74 vs product 0.93; bits3+overlap 1.49 vs product 1.62; top4+overlap 1.87 vs product 2.03; slru+spec2 0.85 vs product 0.91; aware6+overlap 1.62 vs product 1.74; top4+bits3 1.97 vs product 2.08


**Quality-safe (no lossy lever)** (top 8)

| combination | cons | likely | opt | x base | hit | ppl |
|---|---|---|---|---|---|---|
| lru+overlap | 2.89 | 3.14 | 3.06 | 1.26 | 0.56 | +0% |
| pin+overlap | 2.84 | 3.09 | 2.91 | 1.24 | 0.54 | +0% |
| slru+overlap | 2.84 | 3.09 | 2.94 | 1.24 | 0.54 | +0% |
| pin+spec2+overlap | 2.43 | 2.96 | 2.81 | 1.19 | 0.44 | +0% |
| lru+spec2+overlap | 2.43 | 2.94 | 2.80 | 1.18 | 0.44 | +0% |
| slru+spec2+overlap | 2.25 | 2.51 | 2.38 | 1.01 | 0.33 | +0% |
| lru | 2.50 | 2.50 | 2.50 | 1.00 | 0.56 | +0% |
| slru | 2.45 | 2.45 | 2.45 | 0.98 | 0.54 | +0% |

**Measured perplexity loss <= 12%** (top 8)

| combination | cons | likely | opt | x base | hit | ppl |
|---|---|---|---|---|---|---|
| lru+3.0bit+spec4+overlap | 2.29 | 3.85 | 5.34 | 1.54 | 0.51 | +10% |
| slru+top6+overlap | 3.60 | 3.85 | 4.13 | 1.54 | 0.62 | +9% |
| pin+3.0bit+spec4+overlap | 2.25 | 3.80 | 5.03 | 1.52 | 0.48 | +10% |
| slru+3.0bit+overlap | 3.62 | 3.79 | 3.97 | 1.52 | 0.71 | +10% |
| lru+top6+overlap | 3.52 | 3.78 | 4.07 | 1.51 | 0.59 | +9% |
| lru+3.0bit+overlap | 3.54 | 3.72 | 3.92 | 1.49 | 0.68 | +10% |
| pin+top6+overlap | 3.46 | 3.72 | 4.02 | 1.49 | 0.57 | +9% |
| pin+3.0bit+overlap | 3.48 | 3.67 | 3.88 | 1.47 | 0.66 | +10% |

**Any (loss unmeasured or large)** (top 8)

| combination | cons | likely | opt | x base | hit | ppl |
|---|---|---|---|---|---|---|
| slru+top4+3.0bit+spec4+skip25+overlap | 4.82 | 8.08 | 12.94 | 3.24 | 0.81 | ? |
| lru+top4+3.0bit+spec4+skip25+overlap | 4.80 | 8.05 | 12.89 | 3.22 | 0.80 | ? |
| pin+top4+3.0bit+spec4+skip25+overlap | 4.73 | 7.92 | 12.69 | 3.17 | 0.77 | ? |
| slru+top4+3.5bit+spec4+skip25+overlap | 4.59 | 7.70 | 12.34 | 3.09 | 0.75 | ? |
| lru+top4+3.5bit+spec4+skip25+overlap | 4.55 | 7.63 | 12.22 | 3.05 | 0.73 | ? |
| slru+top4+3.0bit+spec4+skip25 | 4.65 | 7.54 | 11.45 | 3.02 | 0.81 | ? |
| lru+top4+3.0bit+spec4+skip25 | 4.62 | 7.49 | 11.34 | 3.00 | 0.80 | ? |
| pin+top4+3.5bit+spec4+skip25+overlap | 4.45 | 7.48 | 11.97 | 2.99 | 0.69 | ? |

Best quality-safe gain over base: x1.26.

## 16 GB machine - base path 4.19 tok/s (likely scenario, mean of 3 traces)

Single levers (x base): slru 1.00, pin 1.00, top6 1.14, top4 1.30, aware6 1.16, bits3.5 1.03, bits3 1.04, spec2 0.98, spec4 1.08, skip25 1.36, overlap 1.04

Pairs: 0 better than the product of their parts (>3%), 0 worse (<-3%), 49 about equal.


**Quality-safe (no lossy lever)** (top 8)

| combination | cons | likely | opt | x base | hit | ppl |
|---|---|---|---|---|---|---|
| slru+spec4+overlap | 2.88 | 4.82 | 7.73 | 1.15 | 0.92 | +0% |
| lru+spec4+overlap | 2.88 | 4.81 | 7.71 | 1.15 | 0.92 | +0% |
| pin+spec4+overlap | 2.87 | 4.80 | 7.70 | 1.15 | 0.92 | +0% |
| slru+spec4 | 2.79 | 4.55 | 6.97 | 1.09 | 0.92 | +0% |
| lru+spec4 | 2.78 | 4.54 | 6.94 | 1.08 | 0.92 | +0% |
| pin+spec4 | 2.78 | 4.52 | 6.91 | 1.08 | 0.92 | +0% |
| slru+overlap | 4.31 | 4.37 | 4.42 | 1.04 | 0.95 | +0% |
| lru+overlap | 4.30 | 4.36 | 4.42 | 1.04 | 0.95 | +0% |

**Measured perplexity loss <= 12%** (top 8)

| combination | cons | likely | opt | x base | hit | ppl |
|---|---|---|---|---|---|---|
| pin+top6+spec4+overlap | 3.27 | 5.48 | 8.78 | 1.31 | 0.93 | +9% |
| lru+top6+spec4+overlap | 3.27 | 5.47 | 8.77 | 1.31 | 0.93 | +9% |
| slru+top6+spec4+overlap | 3.27 | 5.47 | 8.77 | 1.31 | 0.93 | +9% |
| pin+top6+spec4 | 3.19 | 5.22 | 8.04 | 1.24 | 0.93 | +9% |
| lru+top6+spec4 | 3.18 | 5.21 | 8.02 | 1.24 | 0.93 | +9% |
| slru+top6+spec4 | 3.18 | 5.21 | 8.02 | 1.24 | 0.93 | +9% |
| pin+3.0bit+spec4+overlap | 2.99 | 5.00 | 8.01 | 1.19 | 0.94 | +10% |
| lru+3.0bit+spec4+overlap | 2.98 | 4.99 | 7.99 | 1.19 | 0.94 | +10% |

**Any (loss unmeasured or large)** (top 8)

| combination | cons | likely | opt | x base | hit | ppl |
|---|---|---|---|---|---|---|
| pin+top4+3.0bit+spec4+skip25+overlap | 5.10 | 8.53 | 13.66 | 2.03 | 0.92 | ? |
| lru+top4+3.0bit+spec4+skip25+overlap | 5.08 | 8.49 | 13.61 | 2.03 | 0.92 | ? |
| slru+top4+3.0bit+spec4+skip25+overlap | 5.08 | 8.49 | 13.61 | 2.03 | 0.92 | ? |
| pin+top4+3.5bit+spec4+skip25+overlap | 5.06 | 8.47 | 13.56 | 2.02 | 0.92 | ? |
| lru+top4+3.5bit+spec4+skip25+overlap | 5.04 | 8.44 | 13.51 | 2.01 | 0.92 | ? |
| slru+top4+3.5bit+spec4+skip25+overlap | 5.04 | 8.44 | 13.51 | 2.01 | 0.92 | ? |
| pin+top4+spec4+skip25+overlap | 5.01 | 8.37 | 13.41 | 2.00 | 0.92 | ? |
| lru+top4+spec4+skip25+overlap | 4.99 | 8.35 | 13.37 | 1.99 | 0.92 | ? |

Best quality-safe gain over base: x1.15.

## 32 GB machine - base path 4.28 tok/s (likely scenario, mean of 3 traces)

Single levers (x base): slru 1.00, pin 1.01, top6 1.12, top4 1.28, aware6 1.13, bits3.5 1.01, bits3 1.02, spec2 0.99, spec4 1.09, skip25 1.33, overlap 1.03

Pairs: 0 better than the product of their parts (>3%), 0 worse (<-3%), 49 about equal.


**Quality-safe (no lossy lever)** (top 8)

| combination | cons | likely | opt | x base | hit | ppl |
|---|---|---|---|---|---|---|
| pin+spec4+overlap | 2.96 | 4.95 | 7.92 | 1.16 | 0.95 | +0% |
| lru+spec4+overlap | 2.93 | 4.90 | 7.84 | 1.15 | 0.94 | +0% |
| slru+spec4+overlap | 2.93 | 4.90 | 7.84 | 1.15 | 0.94 | +0% |
| pin+spec4 | 2.89 | 4.75 | 7.38 | 1.11 | 0.95 | +0% |
| lru+spec4 | 2.85 | 4.67 | 7.21 | 1.09 | 0.94 | +0% |
| slru+spec4 | 2.85 | 4.67 | 7.21 | 1.09 | 0.94 | +0% |
| pin+overlap | 4.40 | 4.44 | 4.48 | 1.04 | 0.97 | +0% |
| lru+overlap | 4.36 | 4.41 | 4.45 | 1.03 | 0.96 | +0% |

**Measured perplexity loss <= 12%** (top 8)

| combination | cons | likely | opt | x base | hit | ppl |
|---|---|---|---|---|---|---|
| pin+top6+spec4+overlap | 3.32 | 5.55 | 8.89 | 1.30 | 0.94 | +9% |
| lru+top6+spec4+overlap | 3.28 | 5.49 | 8.80 | 1.28 | 0.93 | +9% |
| slru+top6+spec4+overlap | 3.28 | 5.49 | 8.80 | 1.28 | 0.93 | +9% |
| pin+top6+spec4 | 3.25 | 5.34 | 8.29 | 1.25 | 0.94 | +9% |
| lru+top6+spec4 | 3.20 | 5.24 | 8.09 | 1.23 | 0.93 | +9% |
| slru+top6+spec4 | 3.20 | 5.24 | 8.09 | 1.23 | 0.93 | +9% |
| pin+3.0bit+spec4+overlap | 3.01 | 5.04 | 8.07 | 1.18 | 0.95 | +10% |
| lru+3.0bit+spec4+overlap | 2.98 | 4.99 | 7.99 | 1.17 | 0.94 | +10% |

**Any (loss unmeasured or large)** (top 8)

| combination | cons | likely | opt | x base | hit | ppl |
|---|---|---|---|---|---|---|
| pin+top4+3.0bit+spec4+skip25+overlap | 5.15 | 8.60 | 13.78 | 2.01 | 0.94 | ? |
| pin+top4+3.5bit+spec4+skip25+overlap | 5.11 | 8.54 | 13.68 | 2.00 | 0.94 | ? |
| lru+top4+3.0bit+spec4+skip25+overlap | 5.08 | 8.49 | 13.61 | 1.99 | 0.92 | ? |
| slru+top4+3.0bit+spec4+skip25+overlap | 5.08 | 8.49 | 13.61 | 1.99 | 0.92 | ? |
| pin+top4+spec4+skip25+overlap | 5.05 | 8.44 | 13.53 | 1.97 | 0.93 | ? |
| lru+top4+3.5bit+spec4+skip25+overlap | 5.04 | 8.44 | 13.51 | 1.97 | 0.92 | ? |
| slru+top4+3.5bit+spec4+skip25+overlap | 5.04 | 8.44 | 13.51 | 1.97 | 0.92 | ? |
| pin+top4+3.0bit+spec4+skip25 | 5.08 | 8.40 | 13.20 | 1.97 | 0.94 | ? |

Best quality-safe gain over base: x1.16.

## Reading guide
- A gain here is model output on measured hit rates, not a measured tok/s on the 125B.
- Speculation acceptance, overlap share and batch efficiency are estimates (combo_model.py); the cache-aware 6-of-8 rule and layer skipping have no measured quality cost.
- Policy choice (LRU vs segmented LRU vs held-out pinning) moves throughput by under 3%.

