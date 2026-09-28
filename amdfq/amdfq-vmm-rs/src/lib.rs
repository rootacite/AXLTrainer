//! `amdfq-vmm-rs` — the LD_PRELOAD interposer for the HIP allocation gates, Rust side.
//!
//! `hipMalloc` is served by the peralloc route (`peralloc.rs`): a per-request
//! `hipMemAddressReserve`, the block mapped from a handle created for it, and one shared pad granule
//! mapped behind every block, so the slack a bf16 kernel over-reads costs one granule per device
//! instead of one granule of VRAM per live allocation. `hipFree` unmaps and releases the handle, and
//! then gives the VA back too — so an address can be reserved and mapped again — unless
//! `AMDFQ_VA_NEVER_REUSE=1` asks for the pre-fix behaviour, which keeps the span for the process
//! lifetime. With `AMDFQ_POOL_SIZE` set (bytes, off by default, clamped into [16 MiB, 512 MiB]) a
//! request of at most half that size is instead carved out of a pool of that size (`pool.rs`): one
//! `hipMemCreate`, one reserve, one map and one pad granule for the whole pool, and pure bookkeeping
//! per request. A pool whose last block is freed gives its handle and its VA back. Everything the
//! route declines to serve — a runtime without the VMM entry points, a failed reserve/create/map, a
//! device whose state could not be built — is forwarded to the runtime with the caller's size
//! unchanged, and a free is only ever unmapped here if it came out of an extent this crate reserved.
//! A request that would leave driver-reported free VRAM (`mem_info_vram_total - mem_info_vram_used`)
//! below `AMDFQ_VRAM_RESERVE` (bytes, default `0`, i.e. off) is not forwarded: `hipMalloc`
//! returns `hipErrorOutOfMemory` and does not call `hipMemCreate`. The hooked `hipMemGetInfo` reports
//! that same remaining minus the reserve, so the caching allocator sees compositor and RADV occupancy
//! and does not retry sizes this crate will refuse.
//!
//! ```bash
//! cargo build --release --offline                 # -> target/release/libamdfq_vmm_rs.so
//! bash test.sh                                    # build, run one torch allocation, report
//! bash run.sh test | train                        # a torch workload, or a real training run
//! LD_PRELOAD=$PWD/target/release/libamdfq_vmm_rs.so bash ../../start_train.sh
//! ```
//!
//! Like the retired C VMM tree (`amdfq-vmm/`, deleted), this object is never preloaded by `../../start_train.sh` on its own: something
//! that rewrites the process's memory layout does not get to take effect without being asked for.
//!
//! Log lines go to stderr through the `log` crate, one line per call, tagged with the emitting
//! module and the thread id. Every line names what happened: `served` (a block out of our own
//! extent), `forwarded` (the runtime's), `released`, or — at `warn` — the two anomalies worth seeing,
//! an address that was already live (`duplicate`) and a free of an address the registry never held
//! (`untracked`). `AMDFQ_LOG_LEVEL` (`off`/`error`/`warn`/`info`/`debug`/`trace`, default `info`)
//! filters them. `AMDFQ_VRAM_RESERVE` is the driver-counter free floor in bytes (default `0`);
//! `AMDFQ_VA_NEVER_REUSE` (`0`/`1`, default `0`) keeps a freed span's VA for the process lifetime;
//! `AMDFQ_POOL_SIZE` is the pool size in bytes (default `0`, off), which is also the size of one
//! `hipMemCreate` for everything small enough to be carved out of that pool.
//!
//! One thing here is not an allocation gate: on this box's ROCR, any GPU work leaves the runtime's
//! `AsyncEventsLoop` spinning a whole CPU core for the rest of the process, and the only thing that
//! stops it is touching the runtime from a load-time constructor — before torch's own
//! `libtorch_cpu.so` is mapped, which is earlier than any call the application makes. That
//! warm-up is `early.rs`; `AMDFQ_EARLY_HIP` (`0` to switch it off, default on) and `AMDFQ_HIP_LIB`
//! (an explicit runtime path, for when the interpreter's own one is not what you want touched)
//! drive it. It is the crate's only `init`, and DESIGN.md D7 says why it is allowed to be.
//!
//! | module | role |
//! | --- | --- |
//! | `hooks.rs` | the C symbols themselves: `#[unsafe(no_mangle)] pub unsafe extern "C" fn` (`hipMalloc` / `hipFree` / `hipMemGetInfo`) |
//! | `early.rs` | the load-time HIP warm-up that stops the ROCr `AsyncEventsLoop` spin, and where it finds the runtime |
//! | `peralloc.rs` | the VMM route: `serve` / `release`, the per-device state, the mapped-span (or never-reused) VA set |
//! | `pool.rs` | the allocation pool: `PoolSize` from `AMDFQ_POOL_SIZE`, the table of live pools, carving one request out of a pool, giving a pool back when its last block goes |
//! | `real.rs` | the only place a real symbol is resolved: next-object lookup, then cached |
//! | `registry.rs` | `Address -> HookData`, the only shared state over an allocation, and its only lock |
//! | `logging.rs` | the `log` sink (stderr, one `write(2)` per line) |
//! | `hip.rs` | the HIP types and codes this crate names, spelled out so no ROCm headers are needed |
//!
//! The design rules this crate follows — where state lives, how concurrency is treated, what may
//! become a global — are in `DESIGN.md`.

mod early;
mod hip;
mod hooks;
mod logging;
mod peralloc;
mod pool;
mod real;
mod registry;
