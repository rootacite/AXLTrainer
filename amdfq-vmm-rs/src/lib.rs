//! `amdfq-vmm-rs` — the LD_PRELOAD interposer for the HIP allocation gates, Rust side.
//!
//! `hipMalloc` is served by the peralloc route (`peralloc.rs`, ../amdfq.md §12): a per-request
//! `hipMemAddressReserve`, the block mapped from a handle created for it, and one shared pad granule
//! mapped behind every block, so the slack a bf16 kernel over-reads costs one granule for the whole
//! process instead of one granule of VRAM per live allocation. Everything the route declines to serve
//! — a fork child, a runtime without the VMM entry points, a failed reserve/create/map — is forwarded
//! to the runtime with the caller's size unchanged, and a free is only ever unmapped here if it came
//! out of an extent this crate reserved.
//!
//! ```bash
//! cargo build --release --offline                 # -> target/release/libamdfq_vmm_rs.so
//! bash test.sh                                    # build, run one torch allocation, report
//! bash run.sh test | train                        # a torch workload, or a real training run
//! LD_PRELOAD=$PWD/target/release/libamdfq_vmm_rs.so bash ../start_train.sh
//! ```
//!
//! Like `../amdfq-vmm/`, this object is never preloaded by `../start_train.sh` on its own: something
//! that rewrites the process's memory layout does not get to take effect without being asked for.
//!
//! Log lines go to stderr through the `log` crate, one line per call, tagged with the emitting
//! module and the thread id. Every line names what happened: `served` (a block out of our own
//! extent), `forwarded` (the runtime's), `released`, or — at `warn` — the two anomalies worth
//! seeing, an address that was already live (`duplicate`) and a free of an address the registry never
//! held (`untracked`). `AMDFQ_LOG_LEVEL` (`off`/`error`/`warn`/`info`/`debug`/`trace`, default
//! `info`) filters them.
//!
//! | module | role |
//! | --- | --- |
//! | `hooks.rs` | the C symbols themselves: `#[unsafe(no_mangle)] pub unsafe extern "C" fn` |
//! | `peralloc.rs` | the VMM route: `serve` / `release`, the shared pad, the per-process state |
//! | `real.rs` | the only place a real symbol is resolved: next-object lookup, then cached |
//! | `registry.rs` | `Address -> HookData`, the only shared state and the only lock |
//! | `logging.rs` | the `log` sink (stderr, one `write(2)` per line) |
//! | `hip.rs` | the HIP types and codes this crate names, spelled out so no ROCm headers are needed |
//!
//! The design rules this crate follows — where state lives, how concurrency is treated, what may
//! become a global — are in `DESIGN.md`.

mod hip;
mod hooks;
mod logging;
mod peralloc;
mod real;
mod registry;
