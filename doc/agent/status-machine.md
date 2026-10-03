# Status machine — Chromatrix

> Detail behind `AGENT.md` §3. `AGENT.md` keeps the condensed rules; this file carries the full text.


Defined in `trainer/control.py`. Do not add statuses without updating `API.md`, Chromatrix `TrainStatus`, and tests.

```
idle → starting → encoding → training → sampling → finished
                      ↑          ↑          ↑
                   pausing ↔ paused ↔ resuming
                              ↓
                           stopping → finished
dead PID while "live" → error   (reconcile)
reset (PID dead)      → idle    (keeps samples + TB logs; optional weight delete)
```

`LIVE_STATUSES`: starting, encoding, training, sampling, pausing, paused, resuming, stopping.

`is_pid_alive` (`control.py`) reads the process state through `orphans.is_running`, so a **zombie** counts as gone — `os.kill(pid, 0)` alone succeeds on one. `api.py` never `wait()`s the trainer (or a generator) it spawns, so a finished run's trainer stays a zombie until api.py exits (which is when Chromatrix closes), and the old signal-only check told the dashboard `alive: true` for a run that was over: the checkpoint panel's `Generate sample` stayed disabled in that session and only became usable after a restart. Everything that reads a PID (`reconcile`, `train_start`, `_require_alive`, `_gpu_busy`, `_running_generation`, `_reconcile_generated`) goes through it.

`PHASE_STATUSES` (pause/resume attach here): encoding, training, sampling.

Commands (`command.json`): `pause` | `resume` | `stop`. One-shot, sequenced. Trainer consumes them **only** at `device_swap.at_safe_point(phase, swap_ctx)`.

Live settings (`settings.json`): `{save_every_n_steps, sampling_enabled}`, written by `api.py` (`train_settings`, and `train_start` seeding it from `config.toml`), read by the trainer on every optimizer step (`control.read_settings` → `LiveSettings.adopt` in `loop.adopt_live_settings`, after `at_safe_point` so a change made while paused applies to the step the run resumes with). `train_settings` merges onto that file, and a field the file does not carry falls back to the settings the run published (`request_settings(baseline=…)`, which `api.py` fills from `state.json`): a hand-started run (`bash start_train.sh`, or a wiped runtime dir) has no file at all, and without that fallback a lone switch flip would also write the default cadence `0` — silently turning checkpoints off. `state.json` carries the **effective** values in its `settings` block (`{save_every_n_steps, sampling_enabled, next_save_step}`, published by the trainer through `control.publish_settings`), which is what Chromatrix displays, and the `train_status` payload adds a `requested` block whenever `settings.json` asks for something else while a PID is alive (`api._requested_settings`) — the acknowledgement that makes a click answerable at once: a switch flipped during a sample pass shows as the requested value with a "Pending (applies from the next sample pass)" line until the trainer adopts it, instead of the card sitting on the old value for the whole pass. `reset_to_idle` deletes the file, so a change never leaks into the next run. `next_save_step` replaces the old `global_step % N == 0` rule: the first checkpoint is at step N, each save sets `next = step + N`, and a cadence change at step S sets `next = S + N` ("every N steps from now"). Untouched, the sequence is identical to the modulo rule; flipping only the switch leaves the schedule alone. `0` still means "write no checkpoints".

Pause **must** offload the denoise network, all text encoders, both optimizers (including Schedule-Free state), and VAE to CPU, then `empty_cache`. Resume reloads what the current phase needs. Do not “pause” by sleeping on GPU. `SwapContext` uses `denoise` + `text_encoders: list` so a future family can add a third encoder without a new swap shape.

Early-stop semantics (keep these):

| Phase | Stop behavior |
| --- | --- |
| encoding | no LoRA written |
| training | save `{output_name}.safetensors` only if this step has no checkpoint yet |
| sampling | skip leftover repeats (step checkpoint already exists) |

`train_start` fails if a live PID exists, including a process that marked `finished` but has not exited. `train_reset` fails while status is in `_RESET_BLOCKED` **and** PID is alive.

`state.json` also carries `run_id` (the run directory created for this run, written by `control.begin_run`) and `resume` (`null`, or `{path, filename, step, epoch, loaded, skipped}` for a run seeded from a checkpoint via `control.set_resume`).

