# Recipes — Chromatrix

> Detail behind `AGENT.md` §9. `AGENT.md` keeps the condensed rules; this file carries the full text.


### Add an IPC method

1. `handle_*` + `_HANDLERS` in `api.py`.
2. Request/response in `API.md`.
3. `TrainerIpcClient` method + kotlinx.serialization models.
4. A resource row in `IpcResources.kt` (`ranko/shared/src/commonMain/…/data/`): what the method claims and how long it may wait. A method with no row means "no lock at all", and `IpcResourcesTest.thePolicyCoversEveryMethodTheClientSends` reads the client's own `call("…")` literals, so the suite fails until the row is there.
5. Test in `test/test_api_ipc.py` (and Kotlin `DashboardIpcTest.kt` if the payload is parsed).
6. Never print to stdout from the handler.

### Add a base family

1. Catalog row in `trainer/family.py` `CATALOG` **and** Chromatrix `ModelSpecCatalog.kt` (same strings).
2. Implement `trainer/family_<id>.py` (load / unpack / LoRA / encode / loss / save / sample).
3. Register it in `resolve_family`. Set `trainable=True` only when the methods work.
4. Tests in `test_family.py` + Kotlin catalog test.
5. `doc/configuration.md` table.

Do not add a fifth TOML key. `base_model_version` is the dispatch key; the other `[model_spec]` fields must match the catalog row.

### Change training math / sampling

- Family loss / encode: `trainer/family_sdxl.py` (`compute_loss`, `encode_prompts`).
- Loop plumbing: `trainer/loop.py` (`build_group_inputs`, cadence).
- Prompt encoding (SDXL): root `text_processing.py`.
- Sample images: `trainer/sampling.py` via `SdxlFamily.generate_sample`.
- Keep `control.set_training` / `set_sampling` / `set_encoding` in sync so the dashboard progress bars stay honest.

### Change pause/offload

Only `trainer/device_swap.py` + call sites of `at_safe_point`. Iterate `SwapContext.text_encoders`; do not re-hardcode two TEs. Cover with `test_train_control.py` (optimizer tensor moves, command seq, stale PID). Do not skip Schedule-Free optimizer state when offloading.

### Change cleanup targets

Single helper: `trainer/cleanup.py`, always scoped to one run (`run_id`), with `run_id=None` meaning the legacy flat layout. `clean.py` is the interactive CLI (`--run`, `--legacy-flat`) and deletes samples + logs, plus weights when the user confirms; `train_reset` is the API and does nothing when no run resolves — it calls the same helper with all three flags off, because a reset run stays in the Dashboard's history list with its charts, samples and checkpoints. Do not give the API a way to delete weights again: Reset is the button a user reaches for at the end of a run.

### Change resume / checkpoint loading

`ModelFamily.load_lora` (implemented in `family_sdxl.py`) + `trainer/checkpoints.py`. Keep the kohya key map derived from the save-side `_convert_peft_to_kohya_bf16`, cover with `test_family.py` (map, round trip, rank mismatch, zero matches). Weights only — do not promise optimizer-state resume without implementing it.

