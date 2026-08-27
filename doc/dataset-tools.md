# Dataset Tools

Beyond the desktop app, the repo ships several scriptable tools for preparing and cleaning caption datasets, plus a machine-friendly CLI for automation.

## `tools/` — caption/dataset utilities

All scripts live in `tools/` and run from anywhere (paths are positional). Captions are the comma-separated tag lists in the `.txt` files next to images. Unless noted, operations are **destructive in place** — back up before bulk edits.

| Script | Purpose | Usage |
| --- | --- | --- |
| `caper.py` | Remove given tags from every caption in a folder, rewriting files in place. | `python caper.py <path> -r TAG [TAG ...] [-e EXT]` (default ext `.txt`) |
| `dropper.py` | Down-sample a tag: move image+`.txt` pairs whose caption contains a tag into `<dir>/trash` with probability `rate`. Rerun-safe. | `python dropper.py <dir> <tag> <rate>` (rate in `(0, 1]`) |
| `tag_coser.py` | Trash image+`.txt` pairs failing tag conditions: `-n` (trash if ANY negative tag present), `-p` (trash if ANY required tag missing). **Substring** matching. | `python tag_coser.py <dir> [-n TAGS] [-p TAGS]` |
| `tag_counter.py` | Read-only tag frequency report (rank, count, % of files). | `python tag_counter.py <dir>` |
| `tag_filter.py` | List images whose caption contains **all** given tags (AND, case-insensitive). Read-only. | `python tag_filter.py <dir> 'tag1, tag2'` |
| `tag_editor.py` | PyQt6 GUI caption editor (thumbnail list + preview + editor). | `python tag_editor.py <dataset_dir>` |
| `suf.py` | Shuffle dataset order and renumber files to zero-padded sequences, keeping image+caption pairs together. | `python suf.py <directory_path>` |
| `stand_compose.py` | Composite transparent "stand" PNGs onto random background images (scaled to background height, centered). | `python stand_compose.py <stand_dir> <bg_dir> <output_dir>` |
| `inspect_lora.py` | Inspect a LoRA `.safetensors`: metadata dict, key count, prefix distribution (`lora_unet`, `lora_te`, …), sample keys with shapes/dtypes. Read-only. | `python inspect_lora.py <lora.safetensors>` |
| `dumper.py` | Dump a directory tree + all readable file contents into one UTF-8 text file (respects `.dumpignore`, skips binaries, honors `--max-bytes`). Useful for sharing project context with an AI. | `python dumper.py [root] -o OUTPUT [--max-bytes N] [--include-hidden] [--follow-symlinks] [--no-verbose]` |
| `snapping.py` | KDE/Wayland active-window screenshot (2 s delay, then captures and crops the titlebar). Exploratory helper. | `python snapping.py` |

Notes:

- `caper.py`, `dropper.py`, `tag_counter.py`, `tag_filter.py`, and `tag_coser.py` match tags as exact comma-separated tokens (after stripping whitespace) — except `tag_coser.py`, which uses substring matching.
- `dropper.py` / `tag_coser.py` move files to a `trash/` subfolder rather than deleting, so mistakes are recoverable.
- `tag_editor.py` (Qt6) skips anything under a `trash` dir.

## `tagger/` — ONNX caption generator

A WD-tagger-style auto-captioner that writes a `.txt` next to each image in a folder, using an ONNX model on AMD GPU (MIGraphX provider) with CPU fallback.

```bash
cd tagger          # model.onnx and selected_tags.csv are loaded by relative path
python main.py
```

It runs an interactive REPL:

1. Enter a directory path (tab-completion enabled).
2. Enter a confidence threshold (default `0.35`).
3. For every image (png/jpg/jpeg/webp/bmp) it writes `", ".join(tags)` sorted by confidence into `<image>.txt`.

The model is 448×448 input; labels come from `selected_tags.csv`. Type `exit` / `q` to quit. The `migraphx_cache/` folder inside `tagger/` is a compiled-model cache created on first run.

## `ranko/tools/agent.py` — dataset CLI for scripts & AI agents

A non-interactive, machine-friendly CLI that mirrors the desktop app's dataset features (browsing, caption editing, tag statistics, bulk cleanup). It **never reads image pixels** — it only manages the `.txt` caption files next to the images — and is safe to hand to automation.

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
python ranko/tools/agent.py --config ../trainer/config.toml stats
python ranko/tools/agent.py --data-dir ./dataset drop --rate 0.2 --seed 42 --dry-run
python ranko/tools/agent.py --data-dir ./dataset check
```

Safety semantics:

- `stats`, `filter`, `remove-tags`, `add-tag`, and `drop` abort with exit code 1 when orphan caption files exist (mirroring the desktop app), unless `--allow-orphans`.
- `list` / `show` are always lenient.
- `--dry-run` works for `set`, `drop`, `remove-tags`, and `add-tag`.
- Exit codes: `0` success, `1` runtime tool error, `2` argparse usage error. Results → stdout, errors → stderr.
- Requires Python 3.11+ (`tomllib` when using `--config`).
