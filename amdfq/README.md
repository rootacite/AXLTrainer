# amdfq

LD_PRELOAD interposers for HIP allocation on RDNA 4.
`start_train.sh` preloads the release `.so` chosen by `[environment].amdfq` (`none` / `tail` / `vmm`; Chromatrix Utils → **ROCm**). `amdfq-*-rs/run.sh` is a manual override, and `./start_hook.sh` starts any other program under the same preload. Both are opt-in: nothing preloads them unless `[environment].amdfq` says so.

| Path | What |
| --- | --- |
| [`amdfq-vmm-rs/`](amdfq-vmm-rs/) | **The hook** (Rust peralloc VMM). `bash amdfq/amdfq-vmm-rs/run.sh test` / `train`; self-check `bash amdfq/amdfq-vmm-rs/test.sh`. Design constraints: [`amdfq-vmm-rs/DESIGN.md`](amdfq-vmm-rs/DESIGN.md). What `hipPointerGetAttributes` / `hipMemGetInfo` answer under it: [`amdfq-vmm-rs/hip-calls.md`](amdfq-vmm-rs/hip-calls.md). |
| [`amdfq-tail-rs/`](amdfq-tail-rs/) | **The tail hook** (Rust route 2). `hipMalloc` still allocates; one shared page is mapped behind a block the runtime does not already back. `bash amdfq/amdfq-tail-rs/run.sh test` / `train`; self-check `bash amdfq/amdfq-tail-rs/test.sh`. Design constraints: [`amdfq-tail-rs/DESIGN.md`](amdfq-tail-rs/DESIGN.md). |

The VMM hook keeps two pre-fix workarounds as optional switches, both off by default — `amdfq_va_never_reuse` and `amdfq_vram_reserve_gib`, documented in [`../doc/configuration.md`](../doc/configuration.md) and settable from Chromatrix Utils → ROCm.

## Sealed

The C prototypes, the probes, the raw traces, the evidence and the writeups this work came out of are sealed as gpg-encrypted bundles under [`../archive/`](../archive/): the original C tail guard, the VMM/VA and cross-process experiment trees, the from-sysfs and gfx1201 overrun records, and the vendor security report. 涉及负责任披露流程，暂不公开.

That is why the two Rust trees above are the only patch objects here, why the mechanical history the `.so`s were distilled from is not in the tree, and why their design notes no longer cite those documents. `archive/README.md` lists what each bundle holds and how to open it.
