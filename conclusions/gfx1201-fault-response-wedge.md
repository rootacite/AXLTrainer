# The gfx1201 fault response can wedge: an over-read that hangs instead of killing the process

**The observation.** In the boot of 2026-09-16 (kernel `7.2.4-zen2-1`), from somewhere between
16:45 and 17:58 onward, every access to unmapped GPU device memory stopped producing a page-fault
kill. The same commands that had produced a fault line, a ROCr message and a dead process for the
whole day now produced none of the three: the GPU stayed ~100 % busy, the process spun forever in
`hipStreamSynchronize`, and the kernel log said nothing at all. A reboot restored the previous
behaviour — but that reboot also loaded a newer kernel (`7.2.6-zen2-1`), so "the state was reset" and
"the driver build changed" are not separated (§1.4, §3).

**Why this is filed separately from the defect.** [`bf16-kernel-overrun.md`](bf16-kernel-overrun.md)
is about the over-read; this document is about the *response* to it. Nothing above user space changed:
the kernel, its arguments, the placement and the library were bit-identical to the recorded state
(§1.2), and the defect reproduces exactly as recorded after the reboot (§1.4). What broke was the
chain "GPU detects the fault → amdgpu/KFD handles it → the process is told and dies". That chain is
the only thing that makes an over-read *observable* as a FAULT, so when it wedges the harness, the
trainer and the automation all go blind in the same way (§5).

---

## 1. What was observed

### 1.1 The symptom, from the outside

While a reproducer was running, sampled five times over 12 s:

| Signal | Value |
| --- | --- |
| Host process | `State: R (running)`, 100 % CPU, `wchan: 0`, `voluntary_ctxt_switches` constant at **3** |
| GPU | `rocm-smi --showuse` 100 % (desktop idle baseline on this box: 16 %) |
| Kernel log | no new `page fault` / `GCVM_L2…` / `Process …` line |
| ROCr | **nothing** — no `Memory Fault Error`, no `Memory access fault by GPU node-1` |
| Process fate | still alive after **164 s** (killed by hand), no signal, no exit |

`hipStreamSynchronize` calls nothing but the runtime's wait; the harness has no host-side loop after
the launch, and the process never blocked in a syscall, so the spin is ROCr polling a completion
signal the GPU never sets. The access itself was still happening: an instrumented run with mapped
slack (`--pad 4096`) returned the exact reference crc, which only a kernel that reads past its
operand can do.

### 1.2 Same commands, one behaviour changed

Every command below was run in the wedged state; "recorded" is what the same command produced
earlier the same day (`bf16-kernel-overrun.md` §2.4/§2.5, `fixes/hip1/hip1.md` §12.9/§12.10):

| Command | Recorded | In the wedged state |
| --- | --- | --- |
| `launch_kernels … --shape 128 256 64 --pad 0` (fills its tiles exactly) | CLEAN, crc `e4a70c8f` | **CLEAN, crc `e4a70c8f`** |
| `launch_kernels … --shape 1280 64 308 --pad 4096` | CLEAN, crc `4ec0d5e7` | **CLEAN, crc `4ec0d5e7`** |
| `launch_kernels … --shape 1280 64 308 --pad 0` (trainer's shape) | FAULT | **hang** (outer timeout) |
| `launch_kernels … --shape 56 120 24 --pad 0` (`min`) | FAULT | **hang** |
| `fixes/hip1/repro_standalone.sh` (route 1: hipModule, one kernel by name) | process killed | **hang** |
| `fixes/hip1/repro_rocblas.sh` (route 2: hipBLASLt chooses the solution) | process killed | **hang** (90 s) |
| `tools/hip/hip_vmm_tail_read` (unrelated program, deliberate read of an unmapped tail) | process killed | **hang** (60 s) |

Four different programs, two independent routes into hipBLASLt, fresh process/VM/queue each time, all
wedged; the shapes that do *not* touch an unmapped page kept returning the recorded crcs. So the fault
path, not the kernel and not the placement, is what changed.

### 1.3 The timeline in the previous boot

| Time (2026-09-16) | Event |
| --- | --- |
| 12:57 | boot, kernel `7.2.4-zen2-1` |
| 12:57–16:45 | 305 `page fault` lines in this boot; of the records that name their process, `launch_kernels` 248, `ltcheck` 40, `sweep_gemm_shapes` 17 |
| 16:45:19 | **last** fault lines of the boot: `ltcheck`, `RW: 0x1` (a write), `status 0x00841051`, `MORE_FAULTS: 0x1` — the candidate-A/C builds driven through hipBLASLt |
| 16:47–17:57 | no GPU work at all (file edits in the repo and the `axl-*` directories only) |
| 17:58, 18:09–18:12 | the wedge (§1.1, §1.2) — every access that must have faulted produced no line |
| 18:39:00 | shutdown; between 16:45 and here the journal has **no** `amdgpu`/`kfd`/`drm` line other than display-format noise |

Nothing in that timeline proves a trigger. The only thing the ordering says is that the wedge appears
after the day's largest fault load, whose final burst was *write* faults from a patched-library
experiment. Treat "fault storm caused it" as unverified.

### 1.4 Recovery, with a confound

After the reboot, `fixes/hip1/repro_standalone.sh` faults again exactly as recorded: ROCr prints
`Memory Fault Error … faulting addr: 0x7f9ca20c9000 … kernel: Cijk_Ailk_Bjlk_BBS_…_MT64x128x16_…`,
and the process dies (141). But `pacman.log` shows `linux-zen 7.2.4.zen2-1 → 7.2.6.zen2-1` upgraded at
**18:36:22**, 2.5 minutes before the new boot: the previous boot ran the old module, this one runs a
new one. The recovery therefore does not distinguish "transient driver/GPU state, cleared by any
reboot" from "behaviour of that particular driver build".

---

## 2. What is established

| # | Statement | How it is known |
| --- | --- | --- |
| 1 | In that state an access to unmapped device memory wedged the queue instead of killing the process | 4 programs, 2 routes, ≥164 s of spin, GPU 100 %, no log line, no ROCr output (§1.1, §1.2) |
| 2 | The state was **not** in the repo, on disk, or in user space | `fixes/hip1` identical to HEAD; the Tensile family `.co` sha256 `d4def22c…` unchanged; nothing under the env's ROCm directories touched that day; no `HSA_*`/`HIPBLASLT_*`/`PYTORCH_NO_HIP_*` anywhere; `ldd` on the harness shows the env's own `libamdhip64`/`libhsa-runtime64` |
| 3 | The break is in the fault-notification chain, **below** ROCm user space | ROCr was never notified (no message, no death), and the wedge survived across fresh processes, i.e. across any per-process user-space state |
| 4 | It was not a general driver or GPU failure | Queue creation, VMM mapping, `rocm-smi`, and a 4096² bf16 matmul (0.22 s, correct) all worked while the wedge was active |
| 5 | The defect itself is unaffected | The wedged-state runs still returned the *recorded* reference crcs on the shapes that fill their tiles, i.e. kernel, arguments and placement were unchanged (§1.2); after the reboot the same kernel faults again with the same ROCr message naming it (§1.4) |
| 6 | The spin is user-space polling, not a blocked syscall | `State: R`, `wchan: 0`, `voluntary_ctxt_switches` constant over five samples |

## 3. What is not established — the layer question

The wedge is at or below the driver. Two families of explanation fit every measurement equally:

1. **The GPU never escalated the fault.** The wave keeps retrying an access that cannot be
   translated (which is what the 100 % GPU use suggests: an evicted/parked queue would idle), the
   hardware never raises the interrupt, and the driver is simply never called. Nothing is "locked
   up"; it is starved.
2. **The driver received it and did not act or notify.** amdgpu's fault bookkeeping (e.g. a burst
   state that never closes) or KFD's fault/eviction state machine drops every subsequent fault:
   no log, no kill, no notification.

The "queue was evicted and waits for a restore that never comes" sub-case is the least likely of
these: an evicted queue is idle, and the measurement says the GPU was busy.

One earlier candidate was **withdrawn on measurement**: the idea that the boot's *final fault burst*
never closed (leaving `MORE_FAULTS` set forever) looked attractive, but the current, healthy boot
logs the same shape — `MORE_FAULTS: 0x1` records followed by records without a full status block —
so the log shape carries no information here.

The evidence needed to decide was not collected at the time (no probes attached, no interrupt
counting, `debug_evictions` off), and it is gone now: the reboot cleared the driver and GPU state, and
the previous boot's journal has nothing after 16:45. **The past occurrence cannot be attributed to a
layer. The next one can** (§4).

---

## 4. The instrument, calibrated

All of this works on this machine without kernel source: the functions below are in
`/proc/kallsyms`, `bpftrace` and `perf` are installed, and `/proc/interrupts` names the GPU's line
(`162: … IR-PCI-MSI-0000:03:00.0 … amdgpu`).

### 4.1 The healthy signature

`sudo bpftrace -e 'kprobe:amdgpu_ih_process { @ih = count(); } kprobe:gmc_v12_0_process_interrupt
{ @gmc = count(); } kprobe:amdgpu_vm_handle_fault { @vm = count(); }
kprobe:kfd_process_evict_queues { @evict = count(); } kprobe:kfd_process_restore_queues
{ @restore = count(); } kprobe:kfd_evict_process_device { @evict_dev = count(); }
interval:s:4 { print(@ih); print(@gmc); print(@vm); print(@evict); print(@restore); print(@evict_dev); }'`

One real fault (`repro_standalone.sh`) under those probes, 2026-09-16 18:47:

| Probe | Count | Reading |
| --- | --- | --- |
| `amdgpu_ih_process` | 3460 in the first ~7 s (≈500/s) | the IH is busy with other clients (display), so **interrupt counting alone cannot discriminate** — the kprobe is required |
| `gmc_v12_0_process_interrupt` | 22 | the fault reached the driver |
| `kfd_evict_process_device` | 22 | one per fault record |
| `kfd_process_evict_queues` / `kfd_process_restore_queues` | 3 / 2 | the fault is handled through **KFD's queue eviction/restore path** — consistent with `no_queue_eviction_on_vm_fault = 0` ("0 = queue eviction") |
| `amdgpu_vm_handle_fault` | absent (never called) | this fault does **not** go through the generic VM-fault helper |

### 4.2 The decision table for the next occurrence

Run the same probe set against a wedged reproducer and read it:

| Observation while the GPU is ~100 % busy and the process spins | Layer it names |
| --- | --- |
| `gmc` count stays 0 | the GPU/firmware never escalated the fault (driver starved, §3.1) |
| `gmc` > 0 but `kfd_evict`/`kfd_restore` stop progressing | KFD's fault/eviction state machine (kernel module memory) |
| `gmc` and the KFD path both run, the process is still never notified | the notification path (KFD events/SMI → ROCr) |
| only a GPU-only reset restores faulting (step 2 below) | state in the GPU/firmware |
| only a module reload / reboot restores it | state in the driver's kernel memory |

### 4.3 Prepared diagnostics

1. **Before a long run, run the canary** (`bash fixes/hip1/repro_standalone.sh`): a fault that kills
   the process means the path is healthy; a hang means the machine is in this state — kill it (it will
   not exit on its own) and go to step 2. Cost: one real fault, the same cost the campaigns paid all
   day (`fixes/hip1/hip1.md` §9: the card is healthy after faults — no reset, no ring timeout).
2. **If it recurs, try the cheapest reset first**: a GPU-only reset does *not* reload the module, so
   it separates §3.1 from §3.2 — `sudo rocm-smi --gpureset -d 0` (the flag exists in this `rocm-smi`;
   whether the ASIC supports the reset is unknown). With debugfs mounted, the driver's own recovery
   node is the other form: `echo 1 | sudo tee /sys/kernel/debug/dri/0/amdgpu_gpu_recover` — debugfs is
   **not** mounted on this box by default. Either way this tears down every GPU context, so the
   desktop goes with it.
3. **Arm the driver's own logging**: `debug_evictions` is writable at runtime —
   `echo Y | sudo tee /sys/module/amdgpu/parameters/debug_evictions`. This build also carries KFD's
   eviction SMI events (`kfd_smi_event_queue_eviction`, `…_queue_restore`,
   `…_queue_restore_rescheduled`), which is a second observer of the same state machine.
4. **To find the trigger rather than wait for it**: drive a few hundred faults deliberately
   (`launch_kernels`'s ladder over the job list does this) and watch whether the path wedges. If it
   does, §4.2 names the layer on the spot. If it does not, the wedge was specific to the 7.2.4 build
   (§1.4) — that comparison is the reason to keep the old package around.

---

## 5. Consequences for this repo

- **An over-read can now present as a stall, not a crash.** `HSA_STATUS_ERROR_MEMORY_FAULT` in a
  child's log is the marker `test/verify_mask_pipeline.py` retries on and
  `conclusions/bf16-overrun-mitigations.md` §2.3 treats as a restart event; in a wedged state that
  marker never appears.
- **The trainer's own status machine cannot see it.** `control.reconcile` turns a *dead* PID into
  `error`; here the PID is alive and the run stays `training`, with progress frozen and GPU at 100 %.
  A user-facing symptom is "the run hangs at some step, forever, and the dashboard says it is fine".
- **The hip1 harnesses have no per-job timeout**, so a hang blocks the driver process instead of being
  reported as a failure; the runs in §1.2 had to be killed by hand.
- **Nothing recorded in the other documents is invalidated.** The reproducers produced the recorded
  fault and the recorded crcs before and after; the wedge is an environmental state of the fault
  *response*, and every evidence chain that depends on "the read happened" still holds.

## 6. Files

| Path | Role |
| --- | --- |
| `fixes/hip1/repro_standalone.sh`, `repro_rocblas.sh`, `launch_fault_kernel.hip`, `launch_fault_via_rocblas.hip`, `launch_kernels.hip` | The reproducers used in §1.2 and as the §4.3 canary |
| `tools/hip/hip_vmm_tail_read.hip` | The unrelated unmapped-read probe that also wedged |
| `fixes/hip1/hip1.md` §12.9/§12.10, §9 | The recorded fault behaviour and the card-health check after faults |
| [`bf16-kernel-overrun.md`](bf16-kernel-overrun.md) | The defect and its evidence chain (unchanged by this document) |
| [`bf16-overrun-mitigations.md`](bf16-overrun-mitigations.md) | The measures that keep a run alive — all of which assume the fault is still reported |
| `journalctl -k -b -1` (2026-09-16 16:40 → 18:39), `pacman.log` 18:36:22 | The timeline of §1.3 and the kernel-upgrade confound of §1.4 |
