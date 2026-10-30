---
id: M42
type: measurement
status: done
links: [Task check: 30 short questions under the best configurations]
date: 2026-10-30
---
# Tuning I-O threads and pool size for the best configuration

Qwen3-30B, 6 experts anchored + cache bias 1.0, pool+overlap, 9 GB ballast, 64 tokens, 3 rounds, medians: 1 I/O thread 3.213 tok/s (3.213/3.240/2.781); 2 threads 4.372 (4.372/1.297/4.453, one disturbed round); 4 threads 4.842 (4.853/3.711/4.842); pool 800 MB with 2 threads 4.045; pool 2000 MB with 2 threads and adaptive size 3.795. More I/O threads help up to 4 (the I/O threads block in read calls and do not compete for cores); the bias makes a small pool nearly as good as a large one (800 MB keeps about 90 percent of the 1500 MB speed), which matters on 8 GB machines. Default changed to 4 I/O threads on machines with 4 or more cores.
