# Unexplained machine resets under load — what was run, what happened, what the evidence says

This document is self-contained: every number, log excerpt and file timestamp it relies on is quoted
here, not linked. It records an investigation into two hard resets of the development machine on
2026-09-23, the reproduction attempts that followed, and what can and cannot be concluded. The
reproduction itself is `repro_reset.sh` in the repository root, described in §7.

Subject machine: AMD RX 9070 XT 16 GB (gfx1201), 28 threads, 31 GB RAM, Arch Linux, NVMe root on
`/dev/nvme0n1p2`, NFS storage at 192.168.1.50 (`/storage`), this host at 192.168.1.244.

---

## 1. The two resets

Both were **unclean power-level stops**, not shutdowns and not software crashes.

| | reset A | reset B |
| --- | --- | --- |
| Wall time (last surviving record) | 03:28:38 | 04:33:07 |
| Next boot starts | 03:29:20 | 04:33:48 |
| Gap | 42 s | 41 s |
| Kernel of the boot that died | `7.2.7-acite++` | `7.2.7-acite++` |
| Kernel that came up | `7.2.6-zen2-1-zen` (stock; bootloader default was `arch.conf` then) | `7.2.7-acite++` (default had been changed to `acite.conf`) |
| fsck after reboot | `/dev/nvme0n1p2: recovering journal`, then 10+ `Clearing orphaned inode …` lines at 03:29:22–23 | `/dev/nvme0n1p2: recovering journal` at 04:33:48 |
| Kernel lines before the stop | none (last kernel line of that boot: a `x86/split lock detection` warning at 03:15:46) | none (last kernel line of that boot: the same kind of warning at 04:32:04) |

"Recovering journal" plus orphaned inodes is the filesystem saying the machine stopped without
flushing — a power cut or reset, not a clean shutdown (a clean one logs a systemd shutdown sequence,
and the two boots logged none).

**Nothing was reported by the driver or the kernel before either stop.** In the whole of reset B's
boot there are zero `GCVM_L2_PROTECTION_FAULT`, zero `amdgpu: GPU reset`, zero Oops/panic, zero MCE
and zero thermal warnings; the only kernel lines in the last hour are the split-lock warnings from
python processes. The same is true for reset A. Both times, journald also lost its tail (the last
flushed entry is 03:15:46 for A and 04:32:04 for B, i.e. 13 and 1 minutes before the stop), so
"nothing was logged" is certain only for what survived — hence the second-host witness in §5.

### 1.1 What was running at reset A (03:27:31 – 03:28:38)

1. A training run, started detached from a shell at 03:27:31:
   `setsid env PATH=<conda env>/bin:$PATH bash start_train.sh > /tmp/smoke-out.log 2>&1`.
   Its `config.toml` had `train_data_dir = /tmp/lllj-smoke/` (40 symlinked images on tmpfs),
   `epoch = 1`, `save_every_n_steps = 10`, and the allocation patch at
   `amdfq = "vmm"` with `amdfq_vram_reserve_gib = 0` and `amdfq_va_never_reuse = false`.
   The patch's own log confirms it: `vram reserve=0 (disabled)`, `va never reuse=off`.
2. A sealing pipeline, started around 03:28:12, i.e. ~9 s into the sampling pass:
   three times `tar -C <repo> -cf - <paths> | zstd -19 -T0 | gpg --symmetric --cipher-algo AES256`
   over ~120 MB of research material, then `rsync -aR --checksum` of ~125 MB to the NFS mount.

What the filesystem recorded while it ran (the journal's tail is lost, these are not):

| Time | Evidence |
| --- | --- |
| 03:27:31 | run directory `outputs/lllj_20260923_032731/` created |
| 03:27:48 | TensorBoard `events.out.tfevents…` written |
| 03:28:03 | `lllj_s000010/lllj.safetensors`, 237 489 696 B (step 10 checkpoint) |
| 03:28:13 / :22 / :30 | `lllj_samples/lllj_000010_p{0,1,2}_0.png`, each **0 bytes** |
| 03:28:38 | last TensorBoard write |
| 03:28:36 / 03:28:38.99 | the sealing pipeline's first bundle complete (953 543 B); its second bundle present but **0 bytes** |

The three 0-byte PNGs and the 0-byte bundle are the same signature: files created, data still in the
page cache, machine gone. The samples were being written, i.e. the run was in its sampling pass.

Recorded peaks in that window (from the same 1 Hz recorder used later): GPU busy 100 %,
GPU power 377 W, VRAM 13 429 MB, junction 73 °C, package 70 °C, NVMe writes 228 MB/s,
network 50 MB/s, load 2.3, 31 GB RAM with ~28 GB available. No value is near a limit
(junction limit is 110 °C).

### 1.2 What was running at reset B (04:32:35 – 04:33:07)

1. The same training run shape, started 04:32:35 from the repository (mirror of the same config).
2. At 04:32:54, the second the sampling pass began, a heavier version of the same burst:
   decrypt the three archive bundles to a local stage (`gpg --decrypt | zstd -d | tar -x`, 125 MB
   written), re-seal them with three `tar | zstd -19 -T0 | gpg` invocations, then `rsync` 125 MB to
   the NFS mount.

The 1 Hz recorder's last row before the stop (this is the last second the machine was alive):

```
iso      epoch      load1 cpu%  memavail pkg edge junction gpu_W nvme  gpu% vramMB gttMB nvmeW netTx zstd gpg tar rsync python trainer_pid
04:33:07 1790109187 2.25  12.9  16.8     64  48   72       305   44.9  100  8486   39    0.0   0.0   1    2   0   0     7      8249
```

and its last minute shows the shape of the window: GPU 100 % with 305–377 W, VRAM 8 434–13 429 MB,
junction 69–73 °C, package 63–68 °C, CPU 9.8–15.0 %, `zstd=1 gpg=2` (the compression), then at
04:33:02 a 228 MB/s NVMe write burst. The recorder stops at 04:33:07; the next boot is 04:33:48.

Per-step timing from the run's TensorBoard log at reset A, for reference: steps ran 1.4–6.8 s apart
on a machine sitting at load 28–30 from an unrelated CPU stress test (this matches the run being
2–4× slower than the ~1.5 s/step measured when the machine is idle).

---

## 2. The first question: was the training or the tooling at fault?

The allocation patch (`amdfq/amdfq-vmm-rs`, an `LD_PRELOAD` interposer) logs every allocation it
serves and every anomaly it sees. In every run in this investigation — including the two that died —
its log shows **0 warnings, 0 `untracked`, 0 `duplicate`**. For the run that was live at reset B:

```
served=5288 freed=4225 warns=0 untracked=0 duplicate=0
INFO T=8249 amdfq_vmm_rs::peralloc vram reserve=0 (disabled)
INFO T=8249 amdfq_vmm_rs::peralloc va never reuse=off
```

and for the 20-step run that completed earlier on the same day:
`served=12904 freed=11814 distinct VAs=4372 untracked=0 duplicate=0 warn lines=0` (the 12 904
allocations landing on 4 372 distinct addresses is the VA-reuse behaviour the patch has when
`amdfq_va_never_reuse = false`). The kernel logged nothing about the GPU before either stop. So
nothing in the two resets points at the trainer or the patch as a *software* cause.

---

## 3. Reproduction attempts and their outcome

The reproduction target was defined as: the same 40-image training run on tmpfs, with the same burst
fired at the start of the sampling pass. Early attempts had defects, which are listed because they
are why some rows below do not count:

* an attempt whose load was released only after the run had finished (its trigger waited for
  `status == training` while the status during a sampling pass is `sampling`);
* an attempt where the burst ran while the run was back in training (the sampling pass had finished);
* two attempts where the *remote* witness file lived inside a directory the next attempt cleared.

Results (`--mode` names are those of `repro_reset.sh`; "peaks" are the maxima of the 1 Hz recorder
for that round):

| # | what was fired at sampling onset | peaks (GPU % / W / VRAM MB / junction °C / NVMe write) | outcome |
| --- | --- | --- | --- |
| 1 | nothing (control) | 100 / 316 / 13 443 / 73 / 228 MB/s | survived |
| 2 | NFS write only (`rsync` 125 MB) | 100 / 355 / 13 443 / 74 / 228 MB/s | survived |
| 3 | compression only (`tar\|zstd -19 -T0`) | 100 / 317 / 13 299 / 74 / 228 MB/s | survived |
| 4 | AES only (`tar\|zstd -1\|gpg`) | 100 / 339 / 13 443 / 74 / 228 MB/s | survived |
| 5 | local disk write only | 100 / 354 / 13 443 / 75 / 233 MB/s | survived |
| 6 | compression + AES + rsync, one pass (`full`) | 100 / **371** (spike 390) / 13 443 / 74 / 242 MB/s | survived |
| 7 | reset A's exact sequence (`seal1`: 3× seal + rsync) | 100 / 332 / 13 452 / 74 / 245 MB/s | survived |
| 8 | reset B's exact sequence (`seal2`: decrypt + 3× seal + rsync) | 100 / 311 / 13 443 / 74 / **300 MB/s** | survived |
| 9–12 | `seal2` four more times (rate test) | peaks 363 / 334 / 392 / **527** W, junction 73–75 °C | all survived |

Attempts 7–8 restored the two differences the earlier `full` mode had missed: the tmpfs dataset with
a cold latent cache (so the run encodes during the window) and, for `seal2`, the decrypt stage before
the compression. Round 8's NVMe write peak (300 MB/s) is *higher* than in either reset (228 MB/s),
and round 12's GPU power peak (527 W) is higher than anything measured during the resets (377 W).

**So the operation is not a deterministic trigger.** Two resets under this load, eleven rounds of the
same load (six factors, two exact replays, four rate-test rounds) with no reset. An earlier report in
this session that the reset had been "reproduced" is withdrawn: it rested on two same-shaped events,
and the same shape has since run repeatedly without reproducing them.

### 3.1 How a reset actually looks from outside

During reset B the 1 Hz recorder wrote its last row at 04:33:07 and simply stopped; the external
kernel-log witness (§5) had no line about the GPU, and the next boot's fsck reported an unclean
shutdown. There is no error to read *at* the moment of failure — the machine stops mid-second. That
is the shape of a hardware-level cut (power delivery, memory/SoC instability, firmware), not of a
software fault, which would leave a trace somewhere in the kernel log.

---

## 4. What the measurements rule out, and what remains

Ruled out by measurement, for both resets:

* **Thermals** — junction 73 °C peak against a 110 °C limit; package 70 °C; NVMe 47 °C; the enclosure
  is nowhere near throttling, and no thermal kernel message exists.
* **Memory exhaustion** — 28 GB of 31 GB available, no OOM message.
* **Driver or kernel software fault** — no page fault, no GPU reset, no Oops, no MCE.
* **The training stack's own logging** — the patch reported no anomaly at any point.

Remaining, in order of plausibility given "abrupt, unlogged, load-correlated":

1. **Power delivery** — 300–527 W transients on the 12 V rails from the GPU, coinciding with NVMe and
   network bursts, tripping PSU protection (OCP/OPP) or a VRM transient. This is the classic profile
   of an unlogged reset, and the rate-test power peaks climbing to 527 W is consistent with it.
2. **Memory/SoC instability** (EXPO/DOCP profile or marginal RAM) — also produces unlogged resets
   under load, and is cheap to test by running with the JEDEC default profile.
3. **Firmware/ACPI or GPU firmware state** — an intermittent reset path with no log.
4. **Pure intermittency**: the two resets may have needed a state (uptime, thermal soak, a particular
   allocation layout, mains conditions) that the replays did not reproduce — reset A happened after
   ~5 h uptime and reset B after ~15 min, so uptime alone does not explain them.

The two resets and the replays also differ in ways that cannot be restored: at both resets the KDE
Wayland session (kwin_wayland, Xwayland, plasmashell, kitty — four GPU clients holding VRAM) was
running, while all replays ran with the desktop stopped (in tmux), i.e. with less VRAM pressure and
fewer GPU clients. Note the desktop's own exit at 04:24:56 is *not* a fault: that is a session stop,
and the four `amdgpu … VM memory stats for proc <desktop process> … is non-zero when fini` lines at
04:24:56–57 are the teardown of those clients' VMs.

---

## 5. How the evidence was recorded (and how to record it again)

Two recorders ran during the replays, both writing with `fsync` per record:

1. **1 Hz hardware recorder** — one CSV row per second: CPU busy %, load, `MemAvailable`, package and
   max-core temperature, GPU edge/junction/VRAM temperature, GPU power (`power1_average`), NVMe
   temperature, GPU busy %, VRAM and GTT used, NVMe written bytes/s, network bytes/s, counts of
   `zstd`/`gpg`/`tar`/`rsync` processes, and the trainer's PID. Sources are `/proc` and
   `/sys/class/drm/card1/device/*` — read-only, cheap.
2. **Kernel-log witness** — `journalctl -k -f -o short-precise`, every line appended and fsynced,
   mirrored to a **second physical machine** (the NFS server), so a reset here cannot take the record
   with it.

The witness's own limits, learned the hard way: a heartbeat that only fires when a kernel line
arrives cannot date the silence, and a witness file inside a scratch directory that a later round
clears is no witness at all. `repro_reset.sh` writes both recorders' output into the round directory
and takes an optional `--witness-remote DIR` for a second host.

---

## 6. Practical conclusions

* The allocation patch and the trainer are not implicated: no anomaly in their logs, no kernel trace
  at either reset.
* The machine has an intermittent, unlogged hard reset that has been observed only while it is at
  simultaneous full-GPU and disk/network load. Eleven replays of that load did not reproduce it, so it
  cannot be summoned; but there is no observation of it outside that load either.
* Working rule: do not run compression/encryption or large file copies concurrently with training —
  in particular not across the sampling pass, where VRAM and power peak.
* If it happens again, the useful next checks are hardware-side, cheapest first: run the memory at its
  JEDEC default profile for a day; reseat the GPU's power cables (a 12VHPWR/PCIe connector is a
  common source of transient trips); cap the GPU power limit (e.g. 300 → 250 W) and see whether the
  propensity changes; check the board firmware version and whether it records a last-reset reason.

---

## 7. The reproduction script

`repro_reset.sh` (repository root) runs the whole thing:

```bash
AXL_PYTHON=$CONDA_PREFIX/bin/python bash repro_reset.sh   # interpreter with torch/accelerate/tensorboard
bash repro_reset.sh                        # reset B's sequence (decrypt + seal + rsync), one round
bash repro_reset.sh --mode seal1           # reset A's sequence (seal + rsync)
bash repro_reset.sh --mode full            # compression + AES + rsync in one pass
bash repro_reset.sh --mode none            # control: the sampling pass and nothing else
bash repro_reset.sh --mode rsync|zstd|gpg|localwrite   # a single factor
bash repro_reset.sh --repeats 10           # rate test
bash repro_reset.sh --out DIR --witness-remote /path/on/another/machine
```

The script refuses to start if the interpreter cannot import `accelerate`, `torch` and
`tensorboard` (the trainer is launched through `start_train.sh`, which calls a bare `python`, so the
interpreter's directory is what goes on `PATH`; a system python produces a silent no-op round).

What it does per round: builds a 40-image symlinked copy of `[environment].train_data_dir` on
`$TMPDIR` with no latent cache (so the run re-encodes), writes a mirror directory holding symlinks to
`trainer/`, `text_processing.py` and `start_train.sh` plus its own `config.toml` (the repository's
`config.toml` is never edited), starts the run detached the way the dashboard does, watches
`state.json` until the sampling pass is active, fires the configured load, waits for the run to end,
and writes `summary.txt` with that round's peaks.

It writes only under `--out` (default `~/axl-reset-repro/<timestamp>`): `repro.log`, and per round
`recorder.csv`, `kernel.log`, `train.out`, `mirror/`, `summary.txt`, plus whatever the load produced
(scratch copies of the re-compressed material). It never writes to `archive/`; the archive bundles are
only read, as the load's input.

Reading a round: `recorder.csv` is the 1 Hz timeline (its last row is the last second the machine was
alive, if a reset happened), `kernel.log` holds the kernel lines for the same window, and
`train.out` holds the trainer's and the patch's output. A reset looks exactly like reset B: the CSV
stops mid-second with no preceding anomaly, `kernel.log` has no GPU or driver line, and
`journalctl -b 0` after the reboot says `recovering journal`.

Two fidelity notes, kept in the script's header as well: the patch's own environment
(`LD_PRELOAD`, `AMDFQ_*`) is resolved from the repository's `config.toml`, because
`trainer/amdfq_patch.py` derives the repo root from its own file rather than from the working
directory — the mirror only overrides the trainer's own config; and the load's size and shape are
fixed to the sealed material (~125 MB) so rounds stay comparable.
