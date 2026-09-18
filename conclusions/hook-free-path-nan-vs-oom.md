# The hook's free path: without it the failure moves from NaN to OOM

**The observation.** With the peralloc hook (`amdfq-vmm-rs/`) preloaded and the part of its `hipFree`
hook that actually undoes an allocation commented out — so every served extent is reserved, created
and mapped, and then never unmapped, never released and never given back — the run stops failing with
NaN and starts failing with an out-of-memory error instead. Reported by the maintainer on 2026-09-18.

**Status.** Reported, not reproduced or measured here, and the two readings of it are not separated.
The experiment as run is not recorded in this file (§4), so nothing below is evidence about the
mechanism — it is a record of the question and of what would answer it. §6 records the one thing
tried since: the device wait H2 implies was added to the free path, and the failure did not change.

## 1. What was changed and what was seen

`amdfq-vmm-rs/src/hooks.rs`'s `hipFree` normally hands the record to `peralloc::release`, which
unmaps the block, unmaps the pad granule behind it, releases the block's allocation handle and gives
the reserved address range back — in that order, `hipMemUnmap` → `hipMemUnmap` → `hipMemRelease` →
`hipMemAddressFree`, one call each (`amdfq-vmm-rs/src/peralloc.rs`; the rule that the runtime must
never see a served pointer either way is `DESIGN.md` D10). With that call commented out, the extent
stays allocated forever: the route still reserves one address range per `hipMalloc` and still creates
one physical allocation per block, so both address space and VRAM grow with every served request and
nothing shrinks them.

The failure changed from NaN to OOM. Which NaN, at which step, and which OOM error text were not
recorded (§4).

## 2. The two readings

**H1 — NaN first, OOM earlier.** Nothing about the NaN changes; the run simply dies of memory
exhaustion before it reaches the point where the NaN used to appear. Under H1 the free path is
irrelevant to the NaN, and "allocate only" is just a faster way to kill the process. Nothing is
explained, and the NaN cause stays whatever it was.

**H2 — the teardown is the NaN.** Not freeing avoids the NaN because freeing is what produces it: the
runtime's free side runs asynchronously with respect to work still in flight, so a kernel that still
references the block reads memory that has been unmapped, released or handed back, and produces NaN.
(Under H2 the same would be true of the runtime's own `hipFree`; the route would only change how
often, how eagerly and at what granularity the teardown happens.) This is the reading that made the
observation interesting: it would put the fault in the free path rather than in the kernels that
consume the memory.

Both readings are consistent with everything observed so far — H1 needs only that the OOM arrives
before the NaN step, which "never free" makes certain. H2's in-flight half has since been tested and
survived the test (§6).

## 3. What would separate them

1. **Step ordering.** In the normal build, note the step at which the NaN appears. In the
   allocate-only build, note the step at which the OOM appears and what the error is. If the OOM
   arrives earlier, H1 stays alive and the observation says nothing yet.
2. **Take memory pressure out of it.** Re-run the allocate-only build with the workload shrunk
   (`test/torch-test.py --mem-scale`, or a smaller batch / resolution if the failing run is training)
   so that the number of allocations a full run would make cannot exhaust the card. If NaN appears
   there anyway, H1 is dead. If it does not, shrink further until it does — or until the run finishes,
   which would leave H1 standing.
3. **Decompose the free path.** Skip the four teardown calls one at a time instead of all four. Which
   one's absence moves the failure tells apart "the mapping is unmapped under a live kernel"
   (`hipMemUnmap`) from "the handle is released under a live kernel" (`hipMemRelease`) from "the
   address range is handed back and reused" (`hipMemAddressFree`).
4. **Delay rather than skip.** Keep the bookkeeping and defer the teardown by N steps (a queue of
   freed extents, drained later). A live-but-late teardown is the cleanest test of H2: if NaN returns
   only when the delay is short relative to the kernel that reads the block, the teardown is racing
   live work rather than corrupting something at rest. As of §6 a device wait precedes the teardown,
   which turns this item into a control: under that wait a short delay should no longer NaN, and if it
   still does, the wait is not doing what §6 assumes it does.
5. **Ask the log which addresses were freed.** Every hook line already records it: `hipMalloc(…) ->
   served va=… block=… extent=…` and `hipFree(ptr=…) -> released size=…`. If a NaN site can be traced
   to an operand whose address appears in an earlier `released` line, H2 has direct support; if the
   NaN'd buffer is still live (no `released` line for it), the teardown was not involved.
6. **Separate physical exhaustion from address-space exhaustion.** With release skipped, both grow:
   one `hipMemCreate` per block that never comes back, and one reserved range per request that never
   comes back. They fail differently and at different times — a failing `hipMemAddressReserve` makes
   the route serve nothing (the log then shows `forwarded` lines) while a failing `hipMemCreate`
   makes `serve` give up on that request. The current log does not name the failing call or its
   return code, so this needs instrumentation first.

## 4. Gaps in this record

Fill these in before treating the observation as evidence: the exact edit (was `peralloc::release`
commented out, or the whole match arm — and if the arm went, what does a free log now?), the
workload and its command line, whether it was `test/torch-test.py` or a training run, the hook
profile (release/debug — `run.sh --debug` builds the other one), the NaN message and the step it
appeared at, the OOM error text and the step it appeared at, and whether the same run without any
preload still reaches the NaN.

## 5. Related, separately measured (not the same observation)

During the port's own verification, `test/torch-test.py`'s `loop: identical work gives an identical
loss every iteration` check (relative spread ≤ 1e-6 over the 5 loop iterations — the check exists to
catch a compute fault that leaves no trace in the kernel log) failed intermittently with the route
loaded: 5 non-zero spreads in 28 route runs (7e-04 up to 5.3e-2), against 0 in 19 control runs (11
with no preload, 8 with the object preloaded but the route switched off). Serialising the route's
runtime calls behind one mutex did not remove it (2 of 6 runs), and giving each allocation its own
pad granule instead of the shared one did not either (1 of 12). Measured 2026-09-18, by
`LD_PRELOAD=amdfq-vmm-rs/target/release/libamdfq_vmm_rs.so ${AXL_PYTHON:-python3} test/torch-test.py`
and reading that one line.

That is a finite, small spread and not a NaN, so it is a different observation from §1 — whether the
two share a cause is unknown. It is recorded here because both live in the same question: does the
hook's own memory handling, rather than the kernels that use it, change a result?

## 6. The free path's device wait was added, and the failure stayed

H2 names one concrete, missing thing: our teardown never waited for the device, so a block could be
unmapped while a kernel that still read it was in flight — where `ihipFree` waits (`SyncAllStreams`)
before freeing memory it does not own in a pool. `amdfq-vmm-rs/src/peralloc.rs`'s `release` now does
the same: one `hipDeviceSynchronize` after the ownership checks and before the first `hipMemUnmap`
(there is no public "sync all streams"; the device-wide sync is the equivalent). Maintainer-reported
2026-09-18: with the wait in place the failure is unchanged — the NaN and the hang both persist.

So the wait is not the whole cause. It does close the window H2 needs, on the device the free runs
on: a kernel already reading the block can no longer still be in flight when the block goes away.
What H2 has left is everything the wait does not cover — `hipMemRelease` and `hipMemAddressFree` still
run while a kernel may still name the range, a later `hipMemAddressReserve` can hand that range back
out and map someone else's block over it, and the wait drains the queued work of the current device,
not of every device the block was mapped on. The statement above is "the failure persists", which is
all a qualitative NaN-or-hang observation can carry: whether the wait removes some of the failures or
none of them needs a rate, and there is no measurement of one.

The change is kept regardless, because it is what the call it replaces does: taking `hipFree`'s job
over means taking over the wait that comes with it. Keeping it also keeps later experiments clean —
"no wait before the teardown" would otherwise be a variable in each of them.

Not recorded: the run and its command line, whether this was `test/torch-test.py` or a training run,
the step the NaN appears at, which of NaN and hang comes first, whether the hang sits inside HIP or in
Python waiting on it, and whether a run with no preload at all fails the same way.
