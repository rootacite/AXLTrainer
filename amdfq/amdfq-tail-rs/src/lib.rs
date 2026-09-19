//! `amdfq-tail-rs` — the LD_PRELOAD interposer for the HIP allocation gates, tail-guard route.
//!
//! `hipMalloc` is forwarded unchanged, then the tail guard (`tail.rs`, ../doc/amdfq.md §13) maps one
//! shared page at the first granule-aligned address at or after the block's end when the runtime
//! backs nothing there, so a bf16 kernel that over-reads its operand lands in memory that exists.
//! The page is one mapping of a handle created once per device; `hipFree` unmaps it and re-guards
//! the predecessor that ended where the freed block started. The ~0.2 % of allocations whose end
//! page cannot be taken are given back and re-allocated with 16 bytes added — §10.1's pad, for that
//! block alone. `AMDFQ_TAIL=0` turns the guard off so the original step-10 fault can be reproduced.
//!
//! ```bash
//! cargo build --release --offline                 # -> target/release/libamdfq_tail_rs.so
//! bash test.sh                                    # build, run one torch allocation, report
//! bash run.sh test | train                        # a torch workload, or a real training run
//! LD_PRELOAD=$PWD/target/release/libamdfq_tail_rs.so bash ../../start_train.sh
//! ```
//!
//! Like `amdfq/amdfq-vmm-rs/`, this object is never preloaded by `../../start_train.sh` on its own:
//! something that rewrites the process's memory layout does not get to take effect without being asked
//! for.
//!
//! Log lines go to stderr through the `log` crate, one line per call, tagged with the emitting
//! module and the thread id. Every line names what happened: `guarded`, `backed`, `unaligned`,
//! `padded`, `unguarded`, `off`, `released`, or — at `warn` — the two anomalies worth seeing, an
//! address that was already live (`duplicate`) and a free of an address the registry never held
//! (`untracked`). `AMDFQ_LOG_LEVEL` (`off`/`error`/`warn`/`info`/`debug`/`trace`, default `info`)
//! filters them; `AMDFQ_TAIL=0` disables the guard.
//!
//! | module | role |
//! | --- | --- |
//! | `hooks.rs` | the C symbols themselves: `#[unsafe(no_mangle)] pub unsafe extern "C" fn` |
//! | `tail.rs` | the route: after malloc, before/after free, the per-device shared guard handle |
//! | `real.rs` | the only place a real symbol is resolved: next-object lookup, then cached |
//! | `registry.rs` | `Address -> HookData` plus the end-address index, the only shared state and the only lock |
//! | `logging.rs` | the `log` sink (stderr, one `write(2)` per line) |
//! | `hip.rs` | the HIP types and codes this crate names, spelled out so no ROCm headers are needed |
//!
//! The design rules this crate follows — where state lives, how concurrency is treated, what may
//! become a global — are in `DESIGN.md`.

mod hip;
mod hooks;
mod logging;
mod real;
mod registry;
mod tail;
