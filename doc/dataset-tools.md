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
| `inspect_lora.py` | Inspect a LoRA `.safetensors`: metadata dict, key count, prefix distribution (`lora_unet`, `lora_te`, …), sample keys with shapes/dtypes. Read-only. | `python inspect_lora.py <lora.safetensors>` |
| `dumper.py` | Dump a directory tree + all readable file contents into one UTF-8 text file (respects `.dumpignore`, skips binaries, honors `--max-bytes`). Useful for sharing project context with an AI. | `python dumper.py [root] -o OUTPUT [--max-bytes N] [--include-hidden] [--follow-symlinks] [--no-verbose]` |
| `snapping.py` | KDE/Wayland active-window screenshot (2 s delay, then captures and crops the titlebar). Exploratory helper. | `python snapping.py` |
| `gen_prompts.py` | TUI wizard that samples Illustrious XL prompts from repo-root `input_matrix.txt` (character prefix, sfw/nsfw/sex pose pool, clothing exposure groups, chest/belly, expression and eye tags). Does not write quality or rating tags. | `python gen_prompts.py --language chinese` (also `--language english`, `--matrix PATH`, `-o FILE`) |

Notes:

- `caper.py`, `dropper.py`, `tag_counter.py`, `tag_filter.py`, and `tag_coser.py` match tags as exact comma-separated tokens (after stripping whitespace) — except `tag_coser.py`, which uses substring matching.
- `dropper.py` / `tag_coser.py` move files to a `trash/` subfolder rather than deleting, so mistakes are recoverable.
- `tag_editor.py` (Qt6) skips anything under a `trash` dir.
- `mask_blur.py` needs Pillow only (no torch) and writes its sidecars at the training image's own size on purpose: the loader resizes a sidecar back to the source size first, so a differently sized one would return as nearest-neighbour steps instead of a blur. `--radius` is the Gaussian σ in source-image pixels, i.e. `--radius 16` reaches a 2048 px image trained at 1024 as roughly 8 px; at ratio 1 the alpha path is exactly binary, while a 2×–4× downscale already leaves a 2–5 px ramp from the loader's own LANCZOS fit. Values outside 8–32 px still run, with a note on stderr; `--dry-run` reports without writing; the sidecar carries an `axl_mask_blur` PNG text chunk holding the radius, and that marker is what makes a rerun replace its own output. Exit code 1 means at least one image could not be read or written.
- `mask_blur.py` fans the images out over `--jobs` worker processes, defaulting to one per CPU core (`0`); a single image, or `--jobs 1`, runs in-process. The pool preserves the input order, so the report reads the same either way, and one worker holds one decoded image at a time — lower `--jobs` if RAM is tight. Measured on 80 tall 立绘 (~1014×3204): 9.0 s at `--jobs 1` against 0.71 s at the default on a 28-core machine.

## `tagger/` — ONNX caption generator

A WD-tagger-style auto-captioner that writes a `.txt` next to each image in a folder. It prefers AMD GPU via ONNX Runtime **MIGraphX** (CUDA if present) and falls back to CPU. Model files sit next to the script (`tagger/model.onnx`, `tagger/selected_tags.csv`) so you can run it from the repo root. Use the `axl` interpreter (`AXL_PYTHON` or `conda run -n axl`).

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

For every image (`png` / `jpg` / `jpeg` / `webp` / `bmp`, non-recursive; `*.mask.png` skipped) it writes `", ".join(tags)` sorted by confidence into `<stem>.txt`. Input is 448×448 BGR. Ranko’s Utils → Environment **Tag dataset** button calls the same script through the `dataset_tag` IPC method. The `migraphx_cache/` folder inside `tagger/` is a compiled-model cache created on first GPU run.

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
