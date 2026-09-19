# amdfq

LD_PRELOAD interposers for HIP allocation, and the lab notes that produced them.
`start_train.sh` does not preload any of this; you have to ask.

| Path | What |
| --- | --- |
| [`amdfq-vmm-rs/`](amdfq-vmm-rs/) | **The hook** (Rust peralloc VMM). `bash amdfq/amdfq-vmm-rs/run.sh test` / `train`; self-check `bash amdfq/amdfq-vmm-rs/test.sh`. Design constraints: [`amdfq-vmm-rs/DESIGN.md`](amdfq-vmm-rs/DESIGN.md). |
| [`amdfq-tail/`](amdfq-tail/) | Route 2 tail guard (`libamdfq.so`). `bash amdfq/amdfq-tail/hook.sh`. |
| [`doc/`](doc/) | Patch writeups, route comparisons, VMM experiments. |

The C VMM tree (`amdfq-vmm/`) was deleted; peralloc lives in `amdfq-vmm-rs/`.

Kernel-overread notes that are not about the patch stay under [`conclusions/`](../conclusions/) (`bf16-kernel-overrun.md`, `bf16-overrun-mitigations.md`, `gfx1201-fault-response-wedge.md`, `hip-runtime-calls.md`).

## `doc/`

| File | What |
| --- | --- |
| [`amdfq.md`](doc/amdfq.md) | Gates, pad cost, four-round accounts, VMM route 1, tail-guard route 2. |
| [`gfx1201-overread-story.md`](doc/gfx1201-overread-story.md) | Chinese narrative of the same investigation. |
| [`live-data.md`](doc/live-data.md) | Chinese working notes for the four-round measurements. |
| [`eva.md`](doc/eva.md) | Cost comparison: pad-16 vs tail guard vs no patch. |
| [`amdfq-vmm-rs.md`](doc/amdfq-vmm-rs.md) | `hipPointerGetAttributes` / `hipMemGetInfo` under the Rust hook. |
| [`hook-free-path-nan-vs-oom.md`](doc/hook-free-path-nan-vs-oom.md) | Skipping VMM teardown: NaN vs OOM. |
| [`pool-budget-vs-steps.md`](doc/pool-budget-vs-steps.md) | `AMDFQ_POOL` budget vs training steps. |
| [`torch-test-toward-trainer.md`](doc/torch-test-toward-trainer.md) | Ratchet from `test/torch-test.py` toward the trainer crash. |
| [`k91-three-mappings.md`](doc/k91-three-mappings.md) | Lab record: k=91 GEMM, three mappings, pool cut. |
| [`torch-test-ladder/`](doc/torch-test-ladder/) | Stdout extracts for that ratchet. |
