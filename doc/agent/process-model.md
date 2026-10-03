# Process model — AXLTrainer

> Detail behind `AGENT.md` §2. `AGENT.md` keeps the condensed rules; this file carries the full text.


```
Ranko (JVM)  --WebSocket JSON-RPC-->  api.py  --reads/writes-->  config, datasets, TB, samples
                                      |  writes command.json
                                      |  spawns (setsid) bash start_train.sh
                                      v
                         python -u trainer/main.py
                                      |
                                      +--> runtime dir: state.json, command.json, settings.json, train.lock, train.log
                                      +--> logging_dir/{run_id}/          TensorBoard + the run's config.toml
                                      +--> output_dir/{run_id}/{name}_*   checkpoints + samples
```

`run_id` = `{output_name}_{YYYYMMDD_HHMMSS}`, created by `trainer/main.py` through `trainer/runs.py` (`create_run_dirs`). Every run gets its own pair of directories, so a later run (whose step counter restarts at 0) never overwrites an earlier one. `output_name` must be filename-safe — letters and digits (any script), `-`, `_`, `.` — because it becomes the run id **and** the artifact directory names: `validate_output_name` (`trainer/runs.py`) refuses a space or a slash, and it is called from `TrainConfig.__post_init__` (so the trainer and `train_start` refuse such a config at startup) and from `fsrpc.config_save` (so a hand-written one cannot be saved); the Utils form mirrors the same rule and shows `OUTPUT_NAME_HINT`. `api.py` resolves the run for `dashboard` / `list_samples` / `train_reset` / `list_runs` as: explicit `run_id` param → `state.json`'s `run_id` → the newest run directory of an explicitly named `output_name`. A request that names nothing means *the run the trainer is on* and stops at `state.json`: there is deliberately no "newest run directory overall" fallback, which used to make a dashboard with no run recorded show a stopped run's step count and size under its `Current run` heading (a run the trainer never recorded is picked from the history list instead). Passing `run_id` alone is enough: `trainer/runs.py` `run_output_name` recovers the name the run id was built from, which is how a run created under another `output_name` (or one whose `logging_dir` directory is gone) still resolves its `{name}_samples` and weight dirs.

Hard rules:

- Training is **detached**. `api.py` `train_start` uses `start_new_session=True` (`setsid`). Closing Ranko must not kill the run.
- Ranko **never** talks to the GPU. After connect, `commonMain` does not read or write trainer files; it only speaks JSON-RPC. Desktop `jvmMain` may spawn `api.py --websocket` and pick paths with FileKit.
- The trainer is `exec`'d by `start_train.sh`, so that shell's PID and **session** become the trainer's. A GPU fault aborts the trainer from inside HIP (see `doc/troubleshooting.md`) without running Python's `atexit`; its DataLoader forkserver then keeps the workers it forked alive, reparented to init, each holding `/dev/kfd` and ~0.5 GB. `start_train.sh` therefore starts `trainer/orphans.py` first, detached, to reap that session once the trainer is gone — keep it, and keep it unable to touch a session that is not the trainer's.
- Ranko uses `python -u api.py --websocket` (default `127.0.0.1:18765`). LAN bind is `--host 0.0.0.0` plus `--allow-ip` / `AXL_WS_ALLOW`; loopback is always admitted. When neither is set the allowlist is `192.168.0.0/16`. WebSocket is the only control channel; there is no stdin NDJSON fallback. Logs / tracebacks go to stderr. The helper serves **one client session**: the first instance to `hello` owns it, its own later connections join it, and any other client is refused with `CLIENT_BUSY` (who owns it, since when) and closed — so the LAN web companion and the desktop cannot drive the same helper at once, and a second Ranko shows the refusal instead of interleaving calls with the first.
- Working directory for `api.py` and `start_train.sh` is the **repo root** (directory that contains `api.py` and `trainer/`).
- Ranko finds that root by walking up from the executable / `user.dir` until a directory looks like one: `api.py` present, or `config.toml` next to the `trainer/` package (`TrainerRepo.looksLikeRepoRoot`). A lone `config.toml` must not qualify — a stranger's file would otherwise be edited. That walk is desktop bootstrap only.

Runtime dir resolution (same in `trainer/control.py` and `api.py`):

1. `$AXL_RUNTIME_DIR`
2. `$XDG_RUNTIME_DIR/axltrainer`
3. `/tmp/axltrainer-$UID`

Files: `state.json`, `command.json`, `settings.json`, `train.lock`, `train.log`. Tests **must** set `AXL_RUNTIME_DIR` to a temp dir (see `test_train_control.py`).

Interpreter override: Ranko uses `$AXL_PYTHON` if set, else `python3`. Training deps live in the conda env `environment.yml` names — currently `axl` (torch `2.13.0+rocm10.0.0`, HIP `7.15.26333`). There is **no** `requirements.txt`.

