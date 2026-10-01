# Dataset Tools

Beyond the desktop app, the repo ships several scriptable tools for preparing and cleaning caption datasets, plus a machine-friendly CLI for automation.

## `tools/` — caption/dataset utilities

All scripts live in `tools/` and run from anywhere (paths are positional). Captions are the comma-separated tag lists in the `.txt` files next to images. An optional loss mask `{stem}.mask.png` (always PNG, grayscale; white = train, black = ignore) may sit next to the image; if it is absent, a training PNG/WebP with an alpha channel uses that alpha as the mask. Listings skip `*.mask.png` so a sidecar is never treated as a training sample. `dropper.py` / `tag_coser.py` move the sidecar with the image+caption pair. `mask_blur.py` writes such a sidecar from the alpha, which is how a hard, binary silhouette edge is softened without touching the training image. Unless noted, operations are **destructive in place** — back up before bulk edits.

| Script | Purpose | Usage |
| --- | --- | --- |
| `caper.py` | Remove given tags from every caption in a folder, rewriting files in place. | `python caper.py <path> -r TAG [TAG ...] [-e EXT]` (default ext `.txt`) |
| `dropper.py` | Down-sample a tag: move image+`.txt` pairs whose caption contains a tag into `<dir>/trash` with probability `rate`. Rerun-safe. | `python dropper.py <dir> <tag> <rate>` (rate in `(0, 1]`) |
| `tag_coser.py` | Trash image+`.txt` pairs failing tag conditions: `-n` (trash if ANY negative tag present), `-p` (trash if ANY required tag missing). **Substring** matching. | `python tag_coser.py <dir> [-n TAGS] [-p TAGS]` |
| `tag_counter.py` | Read-only tag frequency report (rank, count, % of files). | `python tag_counter.py <dir>` |
| `tag_filter.py` | List images whose caption contains **all** given tags (AND, case-insensitive). Read-only. | `python tag_filter.py <dir> 'tag1, tag2'` |
| `tag_editor.py` | PyQt6 GUI caption editor (thumbnail list + preview + editor). | `python tag_editor.py <dataset_dir>` |
| `suf.py` | Shuffle dataset order and renumber files to zero-padded sequences, keeping image+caption pairs together. **It splits names on the last dot, so `{stem}.mask.png` becomes a group of its own**: a run separates every mask from its image and leaves the masks as bare `<N>.png` files that the trainer then reads as samples (measured: 3 samples + 3 masks → 6 groups). Use the Statistics tab's **Shuffle Dataset** for a mask-safe rename. | `python suf.py <directory_path>` |
| `stand_compose.py` | Composite transparent "stand" PNGs onto random background images (scaled to background height, centered). | `python stand_compose.py <stand_dir> <bg_dir> <output_dir>` |
| `mask_blur.py` | Gaussian-blur each image's alpha channel into a `{stem}.mask.png` sidecar, which then overrides that alpha in training. The training images are never modified; a sidecar this tool did not write (hand-painted in Ranko) is skipped unless `--overwrite`. Blurs in parallel, one worker per CPU core. | `python mask_blur.py DIR [DIR ...] [--radius 16] [--jobs N] [--recursive] [--overwrite] [--dry-run] [-v]` |
| `paint_transfer.py` | PyQt6 GUI: paint a region on one image and copy those pixels into other images of **exactly** the same size — the copy is a straight pixel replacement, with no scaling or alignment step, so a size mismatch is refused before the window opens. Left drag paints, right drag erases, wheel / `[` `]` resize the brush, Ctrl+Z undoes a stroke, Clear empties the mask; the painted area shows as a semi-transparent black overlay while the mask under it is a soft-edged coverage, so a transfer that stops short of an edge blends into the target's own pixels instead of cutting them off. Every target is copied to `/tmp/axlpaint-backup-<timestamp>/` before the first write and all targets are composited in memory first, so a refused or cancelled run leaves the dataset untouched. | `python paint_transfer.py SOURCE TARGET [TARGET ...] [--radius 48]` |
| `inspect_lora.py` | Inspect a LoRA `.safetensors`: metadata dict, key count, prefix distribution (`lora_unet`, `lora_te`, …), sample keys with shapes/dtypes. Read-only. | `python inspect_lora.py <lora.safetensors>` |
| `dumper.py` | Dump a directory tree + all readable file contents into one UTF-8 text file (respects `.dumpignore`, skips binaries, honors `--max-bytes`). Useful for sharing project context with an AI. | `python dumper.py [root] -o OUTPUT [--max-bytes N] [--include-hidden] [--follow-symlinks] [--no-verbose]` |
| `snapping.py` | KDE/Wayland active-window screenshot (2 s delay, then captures and crops the titlebar). Exploratory helper. | `python snapping.py` |
| `gen_prompts.py` | **Legacy, frozen — do not edit.** The Kotlin port in Ranko's Automation tab is the maintained one. Its parser predates `QUESTIONABLE_POSES`, `FIGURE`, `PUSSY_SHAPE` and `PUSSY_HAIR` and the newer sampling rules, so those headers are not recognized as sections: hand-running it puts every row under them into the SFW pool, plus one bogus tag row per header (`QUESTIONABLE_POSES:`, `PUSSY_SHAPE:`, …). Generate prompts from Ranko → Automation → Prompts instead. Originally: a TUI wizard that samples Illustrious XL prompts from repo-root `input_matrix.txt` (character prefix, sfw/nsfw/sex pose pool, clothing exposure groups, chest/belly, face tags). The face page is one screen of groups — expression, gaze, eye state, mouth, blush, tears — and each group is a multi-select candidate list: one of the ticked tags reaches a prompt and never two, ticking nothing makes the group write nothing, and `Any` means the group's whole mode pool. A group whose tag a pose already carries is left alone. Sex mode also has a stage-weight page (posing / before / during / object insertion / fingering / ejaculation / after / done) after the vaginal-ratio page; zero skips a stage, and a missing profile key keeps the old during-only behaviour. Object insertion and fingering only attach to poses that allow anal or vaginal, write the matching tag (`anal object insertion` / `vaginal object insertion` / `anal fingering` / `fingering`), and omit penis and the other stage tags. Oral / paizuri / nursing-handjob rows in `POSES` use channel `none` (no hole, no `sex`/`vaginal`/`anal`). Does not write quality or rating tags. Its confirm page can save the finished configuration as a named profile, and the next run offers saved profiles up front: picking one opens a one-page configuration list — every item (`character`, `mode`, exposure, clothing, chest/belly, face, scene, pose families, vaginal share, stage weights, poses, count) with its current value — where `Enter` edits just that row and the bottom rows generate, save as a profile, or go back. `--profile NAME` skips the list and generates straight away. Profiles are JSON under repo-root `prompt_profiles/` (local, gitignored); profiles written before the grouped face page (format v1) are upgraded on load. | `python gen_prompts.py --language chinese` (also `--language english`, `--matrix PATH`, `-o FILE`, `--profile NAME`, `--profiles-dir DIR`) |

The prompt wizard and the ComfyUI batch both live in Ranko now (Automation tab: the wizard, the
named profiles and the generated list on the left, the batch and the gallery next to it — see
[dashboard.md](dashboard.md#automation--prompt-wizard-comfyui-batch-gallery)). The two scripts stay
as the headless entry points, and they are still the reference implementation for anything the port
does not cover; a job's images land under `automation/jobs/` when started from the GUI.

Notes:

- `caper.py`, `dropper.py`, `tag_counter.py`, `tag_filter.py`, and `tag_coser.py` match tags as exact comma-separated tokens (after stripping whitespace) — except `tag_coser.py`, which uses substring matching.
- `dropper.py` / `tag_coser.py` move files to a `trash/` subfolder rather than deleting, so mistakes are recoverable.
- `tag_editor.py` (Qt6) skips anything under a `trash` dir.
- `paint_transfer.py` needs PyQt6, Pillow and numpy (all in the `axl` env). A JPEG target is re-encoded with its own quantization tables, chroma sampling and progressive flag (`quality="keep"`, whose path in Pillow requires the image being saved to carry `format="JPEG"`), a WebP target at quality 95, a PNG losslessly; EXIF and ICC profiles are carried over. A palette (`P`) target comes back as RGB and a grey+alpha (`LA`) one as RGBA, since those are the modes the copy is made in. Exit code 1 means a size mismatch, an unreadable image, nothing painted, or a target that could not be written — a write failure is reported per file, with the backups already in place.
- `mask_blur.py` needs Pillow only (no torch) and writes its sidecars at the training image's own size on purpose: the loader resizes a sidecar back to the source size first, so a differently sized one would return as nearest-neighbour steps instead of a blur. `--radius` is the Gaussian σ in source-image pixels, i.e. `--radius 16` reaches a 2048 px image trained at 1024 as roughly 8 px; at ratio 1 the alpha path is exactly binary, while a 2×–4× downscale already leaves a 2–5 px ramp from the loader's own LANCZOS fit. Values outside 8–32 px still run, with a note on stderr; `--dry-run` reports without writing; the sidecar carries an `axl_mask_blur` PNG text chunk holding the radius, and that marker is what makes a rerun replace its own output. Exit code 1 means at least one image could not be read or written.
- `mask_blur.py` fans the images out over `--jobs` worker processes, defaulting to one per CPU core (`0`); a single image, or `--jobs 1`, runs in-process. The pool preserves the input order, so the report reads the same either way, and one worker holds one decoded image at a time — lower `--jobs` if RAM is tight. Measured on 80 tall 立绘 (~1014×3204): 9.0 s at `--jobs 1` against 0.71 s at the default on a 28-core machine.

## `tagger2/` — Pixai tagger v1 caption generator

The captioner Ranko's Utils → Environment **Tag dataset** button runs (IPC `dataset_tag`). It writes a `.txt` next to each image in a folder, the way the legacy script below does, but from a far larger model: `pixai-labs/pixai-tagger-v1.0`, a 1008×1008 ViTDet with 30 877 Danbooru tags in six categories, loaded through its own `TaggerPipeline` (`trust_remote_code`) on PyTorch/ROCm. Use the `axl` interpreter (`AXL_PYTHON` or `conda run -n axl`).

```bash
# Non-interactive (overwrites sidecar captions). Progress on stderr; optional JSON on stdout.
python tagger2/main.py /path/to/dataset --threshold 0.35 --json

# Interactive REPL (directory + threshold prompts, tab-completion; the model loads once)
python tagger2/main.py

# What the model can write: categories, tag counts, calibrated thresholds (no GPU, no network)
python tagger2/main.py --info
```

| Flag | Meaning |
| --- | --- |
| `-t` / `--threshold` | Lowest confidence a tag may keep, `0.0`–`1.0` (default `0.35`). The model's own per-category calibrated threshold is the floor underneath it, so a tag is kept when its probability is at least `max(calibrated, threshold)`. |
| `--categories` | Comma-separated categories to write into the caption (default `general`). Named in the model's own order: `general`, `character`, `copyright`, `style`, `meta`, `rating`. The provenance categories are off by default: 8308 character names, 4917 artist/style tags and `highres`-style meta tags describe where a picture comes from rather than what is in it. `rating` writes the model's own `rating:g` / `rating:s` / `rating:q` / `rating:e`. |
| `--only-tags` | **Partial tagging**: a comma-separated list of tags to add to the captions that show them. `--categories` does not take part (a requested tag is looked up in all six) and `--threshold` becomes the absolute confidence floor for these tags instead of a floor under the calibrated values. |
| `-b` / `--batch-size` | Images per forward pass (default `1`). Measured on an RX 9070 XT: a batch of 3 and three single passes both take 1.60 s warm, so the default stays 1. |
| `--json` | Print one result object to stdout (progress stays on stderr) |
| `--cpu` | Run on the CPU. Without it, `device` is left to transformers, which takes device 0 when the platform offers one — on this stack the GPU. |
| `--model` | Hugging Face id or a local directory (default `pixai-labs/pixai-tagger-v1.0`) |
| `--download` | Allow the one-time Hub fetch (~1.9 GB). Off by default: without it the script is cache-only. |
| `--info` | Print categories, tag counts and calibrated thresholds as JSON, then exit |

For every image (`png` / `jpg` / `jpeg` / `webp` / `bmp`, non-recursive; `*.mask.png` skipped) it writes `", ".join(tags)` sorted by confidence into `<stem>.txt`, with the model's underscores turned into spaces (`hair_between_eyes` → `hair between eyes`, `masking_tape_(medium)` → `masking tape (medium)`) so captions match the ones the legacy tagger wrote. The Chinese names in Statistics → Tags come from the legacy tagger's `selected_tags.csv`, which covers 10 507 of these 30 877 tags (34 %); the rest show in English.

**Partial tagging (`--only-tags`) corrects captions instead of replacing them.** It is the pass a dataset with hand-written or existing captions wants: only the tags you name are looked for, and a tag the model reports above `--threshold` is appended to the caption that lacks it. Every other tag — and the spelling of the ones already there — is left alone, so nothing the tagger wrote last week or you wrote by hand is lost. A caption that would gain nothing is never rewritten, and an image with no sidecar gains one only when something matched. Because the tags are your choice, `--threshold` is handed to the pipeline as its `threshold` (the whole floor) rather than as `min_threshold`: a calibrated value above your request would otherwise hide the very tag you asked for. The JSON result carries `mode: "partial"`, `only_tags`, `added` (how many images gained each tag) and `unmatched` (tags that matched no image, which is usually a spelling the model does not use).

**Never online, and never on a cold MIOpen.** The Hub is not asked at any point: the script resolves the model's snapshot itself (cache-only) and hands the pipeline that directory, which is what keeps the tokenizer/processor/feature-extractor lookups `pipeline()` makes — all of which it can live without — off the network. Measured over three images: 10.9 s and **zero outbound connections**. The repository id, by contrast, cost 16.80 s and one connection to the configured proxy, and stalls where no proxy is set. `--download` is the escape hatch for the first fetch, and the failure message says so.

`MIOPEN_USER_DB_PATH` / `MIOPEN_CUSTOM_CACHE_DIR` are pointed at `tagger2/miopen_cache/` (gitignored, `AXL_TAGGER_CACHE_DIR` overrides, a shell that already pinned them wins) *before* torch is imported, so the first convolutions are not re-searched every start: a cold directory costs about 0.8 s of the first inference (96 KB of find-db written), a warm one keeps it. Warm throughput is ~0.53 s/image after a ~1.3 s construct and a 2.8 s first image, so a 640-image folder is roughly six minutes.

## `tagger/` — WD14 ONNX caption generator (legacy)

Kept as it was: a WD-tagger-style ONNX captioner that prefers AMD GPU via ONNX Runtime **MIGraphX** (CUDA if present) and falls back to CPU. It is no longer reachable from Ranko — `dataset_tag` runs `tagger2/` — but it still runs by hand, and its `tagger/selected_tags.csv` is what `tag_lexicon` reads for the English → Chinese tag names (that file is local, gitignored, and has no equivalent for the new vocabulary). Model files sit next to the script (`tagger/model.onnx`, `tagger/selected_tags.csv`).

```bash
# Non-interactive (overwrites sidecar captions). Progress on stderr; optional JSON on stdout.
python tagger/main.py /path/to/dataset --threshold 0.35 --json

# Interactive REPL (directory + threshold prompts, tab-completion)
python tagger/main.py
```

| Flag | Meaning |
| --- | --- |
| `-t` / `--threshold` | Minimum confidence, `0.0`–`1.0` (default `0.35`) |
| `-b` / `--batch-size` | ONNX batch size (default `1`; MIGraphX compiles per input shape) |
| `--json` | Print one result object to stdout |
| `--cpu` | Force `CPUExecutionProvider` |

For every image (`png` / `jpg` / `jpeg` / `webp` / `bmp`, non-recursive; `*.mask.png` skipped) it writes `", ".join(tags)` sorted by confidence into `<stem>.txt`. Input is 448×448 BGR. The `migraphx_cache/` folder inside `tagger/` is a compiled-model cache created on first GPU run.

## `ranko/tools/agent.py` — dataset CLI for scripts & AI agents

## `ranko/tools/agent.py` — dataset CLI for scripts & AI agents

A non-interactive, machine-friendly CLI that mirrors the desktop app's dataset features (browsing, caption editing, tag statistics, bulk cleanup). It **never reads image pixels** — it only manages the `.txt` caption files next to the images — and is safe to hand to automation. `list` / `scan` skip `*.mask.png`. `drop` moves `{stem}.mask.png` with the image+caption pair when present.

```bash
# Global options (before or after the subcommand):
#   --data-dir DIR | --config PATH   (required; --config reads [environment].train_data_dir)
#   --format json|text               (default json)
#   --dry-run                        (preview mutations without writing)
#   --allow-orphans                  (don't abort when orphan captions exist)
```

| Command | Args | What it does |
| --- | --- | --- |
| `list` | `--limit N` | List all samples (name + tags). Always lenient. |
| `show` | `--name NAME` | Show one sample's caption + parsed tags. |
| `set` | `--name NAME` + `--caption TEXT` XOR `--file PATH` | Write a caption to the `.txt`, creating it if missing (verbatim). |
| `stats` | `--min-count N`, `--limit N` | Tag frequency (counted once per file), sorted desc. |
| `filter` | `--tags TAGS`, `--mode and\|or` | List samples matching all / any tags. |
| `remove-tags` | `--tags TAGS`, `--only TAGS`, `--mode` | Bulk-remove tags from all (or `--only`-filtered) samples; rewrites only changed files in normalized `", "` format. |
| `add-tag` | `--tag TAG`, `--position start\|end`, `--only TAGS`, `--mode` | Bulk-add one tag; never duplicates an existing tag. |
| `drop` | `--rate R`, `--only TAGS`, `--mode`, `--trash-dir DIR`, `--seed N` | Randomly move image+caption pairs to trash (default `/tmp/axlranko/trash`). |
| `check` | — | Integrity report: orphan captions and images without captions; exits 1 if orphans exist. |

Examples:

```bash
python ranko/tools/agent.py --data-dir ./dataset list --limit 10
python ranko/tools/agent.py --data-dir ./dataset stats --limit 20
python ranko/tools/agent.py --data-dir ./dataset filter --tags "solo, 1girl" --mode and
python ranko/tools/agent.py --data-dir ./dataset remove-tags --tags "blurry" --only "solo" --dry-run
python ranko/tools/agent.py --config ../config.toml stats
python ranko/tools/agent.py --data-dir ./dataset drop --rate 0.2 --seed 42 --dry-run
python ranko/tools/agent.py --data-dir ./dataset check
```

Safety semantics:

- `stats`, `filter`, `remove-tags`, `add-tag`, and `drop` abort with exit code 1 when orphan caption files exist (mirroring the desktop app), unless `--allow-orphans`.
- `list` / `show` are always lenient.
- `--dry-run` works for `set`, `drop`, `remove-tags`, and `add-tag`.
- Exit codes: `0` success, `1` runtime tool error, `2` argparse usage error. Results → stdout, errors → stderr.
- Requires Python 3.11+ (`tomllib` when using `--config`).
