# Page-fault run matrix of the verification session (2026-09-14)

Six runs faulted, 7 runs of the same configuration except `seed` completed 120/120 steps.
Every fault happened at step 101 (epoch 51), ~2 min in, right after the step-90 sample generation.

| session | kernel time | pid | run | seed | outcome |
| --- | --- | --- | --- | --- | --- |
| session 1 | 20:14:05 | 214146 | train_m1145141920 | 1145141920 | FAULT at step 101/120 |
| session 1 | 20:17:31 | 214881 | train_c1145141920 | 1145141920 | FAULT at step 101/120 |
| session 1 | 20:20:47 | 217578 | train_g1145141920 | 1145141920 | FAULT at step 101/120 |
| session 2 | 20:53:35 | 231641 | train_m1145141920 | 1145141920 | FAULT at step 101/120 |
| session 3 | 20:59:18 | 234258 | train_m1145141920 (attempt 1) | 1145141920 | FAULT at step 101/120 |
| session 3 | 21:02:27 | 234857 | train_m1145141920 (attempt 2) | 1145141920 | FAULT at step 101/120 |

The surviving runs used `seed = 1145141919` (three in session 1, three in session 2, and the
duplicate-run floor in session 4); the faulting runs used `seed = 1145141920`, which changes the
caption-shuffle RNG and therefore the encoder sequence length and the LoRA GEMM's `M`.
