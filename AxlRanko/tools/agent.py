#!/usr/bin/env python3
"""AxlRanko dataset management CLI for AI agents.

Exposes the core functionality of the AxlRanko desktop app (dataset browsing,
caption editing, tag statistics, and bulk cleanup) as a non-interactive,
machine-friendly command-line tool. It never reads or decodes image pixels;
it only manages the caption (.txt) files next to the images.
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}
DEFAULT_TRASH_DIR = "/tmp/axlranko/trash"


class ToolError(Exception):
    """Runtime error that aborts the command with exit code 1."""


# --------------------------------------------------------------------------
# Dataset scanning
# --------------------------------------------------------------------------

def parse_tags(content):
    """Split a caption into tags: split on commas, trim whitespace, drop empties.

    Mirrors the Kotlin implementation (split(",") -> trim() -> filter notEmpty).
    Duplicates are preserved here; deduplication only happens in statistics.
    """
    return [t.strip() for t in content.split(",") if t.strip()]


def scan_dataset(data_dir, strict=False):
    """Scan a dataset directory (non-recursive, like the desktop app).

    Returns (samples, orphan_txt) where samples is a list of dicts with keys
    name / image / txt / caption / tags. When strict is True and an orphan
    caption file (a .txt with no matching image) exists, raises ToolError,
    mirroring the desktop app's dataset integrity check.
    """
    image_files = []
    txt_by_stem = {}
    for f in sorted(data_dir.iterdir()):
        if not f.is_file():
            continue
        ext = f.suffix.lower().lstrip(".")
        if ext in IMAGE_EXTENSIONS:
            image_files.append(f)
        elif ext == "txt":
            txt_by_stem[f.stem] = f

    image_stems = {f.stem for f in image_files}
    orphan_txt = [txt_by_stem[stem] for stem in sorted(txt_by_stem) if stem not in image_stems]

    if strict and orphan_txt:
        names = ", ".join(p.name for p in orphan_txt)
        raise ToolError(
            "Dataset error: found isolated caption file(s) without a matching image: "
            f"{names}. Execution aborted (mirrors the desktop app's integrity check). "
            "Run `check` to diagnose, or pass --allow-orphans to inspect anyway."
        )

    samples = []
    for im in image_files:
        txt = txt_by_stem.get(im.stem)
        caption = txt.read_text(encoding="utf-8") if txt else ""
        samples.append({
            "name": im.stem,
            "image": str(im),
            "txt": str(txt) if txt else None,
            "caption": caption,
            "tags": parse_tags(caption),
        })
    return samples, [str(p) for p in orphan_txt]


def find_by_name(samples, name):
    """Find a sample by image/caption file stem, tolerating extensions."""
    key = name.strip()
    if key.lower().endswith(".txt"):
        key = key[:-4]
    else:
        for ext in IMAGE_EXTENSIONS:
            suffix = "." + ext
            if key.lower().endswith(suffix):
                key = key[: -len(suffix)]
                break
    for s in samples:
        if s["name"] == key:
            return s
    return None


def parse_tag_list(text):
    return [t.strip() for t in text.split(",") if t.strip()]


def matches(sample, tags, mode):
    """AND: all tags present; OR: any tag present (mirrors the desktop app)."""
    have = set(sample["tags"])
    if mode == "and":
        return all(t in have for t in tags)
    return any(t in have for t in tags)


def resolve_targets(samples, only_tags, mode):
    if not only_tags:
        return samples
    return [s for s in samples if matches(s, only_tags, mode)]


def txt_path_of(sample):
    return Path(sample["txt"]) if sample["txt"] else Path(sample["image"]).with_suffix(".txt")


def _move_replace(src, dest_dir):
    """Move src into dest_dir, replacing an existing file of the same name
    (equivalent to Files.move with REPLACE_EXISTING in the desktop app)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.exists():
        dest.unlink()
    return shutil.move(str(src), str(dest))


# --------------------------------------------------------------------------
# Command implementations
# --------------------------------------------------------------------------

def cmd_list(args, ctx):
    samples, _ = scan_dataset(ctx["data_dir"], strict=False)  # like the Images page
    items = samples
    if args.limit is not None:
        items = items[: args.limit]
    return {"command": "list", "data_dir": str(ctx["data_dir"]), "total": len(samples), "items": items}


def cmd_show(args, ctx):
    samples, _ = scan_dataset(ctx["data_dir"], strict=False)
    s = find_by_name(samples, args.name)
    if s is None:
        raise ToolError(f"Sample not found: {args.name}. Use `list` to see available samples.")
    return {"command": "show", **{k: s[k] for k in ("name", "image", "txt", "caption", "tags")}}


def cmd_set(args, ctx):
    samples, _ = scan_dataset(ctx["data_dir"], strict=False)
    s = find_by_name(samples, args.name)
    if s is None:
        raise ToolError(f"Sample not found: {args.name}. Use `list` to see available samples.")
    if args.caption is not None and args.file is not None:
        raise ToolError("Provide only one of --caption or --file.")
    if args.caption is not None:
        text = args.caption
    elif args.file is not None:
        try:
            text = Path(args.file).read_text(encoding="utf-8")
        except OSError as e:
            raise ToolError(f"Failed to read caption file {args.file}: {e}")
    else:
        raise ToolError("Provide --caption TEXT or --file PATH.")
    txt = txt_path_of(s)
    created = not txt.exists()
    if not ctx["dry_run"]:
        txt.write_text(text, encoding="utf-8")
    return {"command": "set", "name": s["name"], "txt": str(txt), "caption": text,
            "created": created, "dry_run": ctx["dry_run"]}


def cmd_stats(args, ctx):
    samples, _ = scan_dataset(ctx["data_dir"], strict=not ctx["allow_orphans"])
    counter = {}
    for s in samples:
        # Count each tag once per file, mirroring the desktop app.
        for t in set(s["tags"]):
            counter[t] = counter.get(t, 0) + 1
    total = len(samples)
    stats = [
        {"tag": t, "count": c, "frequency": (c / total * 100.0) if total else 0.0}
        for t, c in counter.items()
    ]
    stats.sort(key=lambda x: (-x["count"], x["tag"]))
    if args.min_count is not None:
        stats = [x for x in stats if x["count"] >= args.min_count]
    if args.limit is not None:
        stats = stats[: args.limit]
    return {"command": "stats", "data_dir": str(ctx["data_dir"]), "total_samples": total, "stats": stats}


def cmd_filter(args, ctx):
    tags = parse_tag_list(args.tags)
    if not tags:
        raise ToolError("--tags requires at least one tag.")
    samples, _ = scan_dataset(ctx["data_dir"], strict=not ctx["allow_orphans"])
    items = [s for s in samples if matches(s, tags, args.mode)]
    return {"command": "filter", "data_dir": str(ctx["data_dir"]), "tags": tags,
            "mode": args.mode, "total": len(items), "items": items}


def cmd_remove_tags(args, ctx):
    remove_set = set(parse_tag_list(args.tags))
    if not remove_set:
        raise ToolError("--tags requires at least one tag.")
    samples, _ = scan_dataset(ctx["data_dir"], strict=not ctx["allow_orphans"])
    targets = resolve_targets(samples, parse_tag_list(args.only), args.mode) if args.only else samples
    results = []
    changed = 0
    for s in targets:
        removed = [t for t in s["tags"] if t in remove_set]
        if not removed:
            continue  # skip no-op rewrites
        new_tags = [t for t in s["tags"] if t not in remove_set]
        txt = txt_path_of(s)
        if not ctx["dry_run"]:
            txt.write_text(", ".join(new_tags), encoding="utf-8")
        changed += 1
        results.append({"name": s["name"], "txt": str(txt), "removed_tags": removed, "new_tags": new_tags})
    return {"command": "remove-tags", "data_dir": str(ctx["data_dir"]), "tags": sorted(remove_set),
            "scanned": len(targets), "changed": changed, "dry_run": ctx["dry_run"], "results": results}


def cmd_add_tag(args, ctx):
    tag = args.tag.strip()
    if not tag:
        raise ToolError("--tag must not be empty.")
    samples, _ = scan_dataset(ctx["data_dir"], strict=not ctx["allow_orphans"])
    targets = resolve_targets(samples, parse_tag_list(args.only), args.mode) if args.only else samples
    results = []
    changed = 0
    skipped = 0
    for s in targets:
        if tag in s["tags"]:
            skipped += 1  # never duplicate an existing tag
            continue
        new_tags = ([tag] + s["tags"]) if args.position == "start" else (s["tags"] + [tag])
        txt = txt_path_of(s)
        if not ctx["dry_run"]:
            txt.write_text(", ".join(new_tags), encoding="utf-8")
        changed += 1
        results.append({"name": s["name"], "txt": str(txt), "new_tags": new_tags})
    return {"command": "add-tag", "data_dir": str(ctx["data_dir"]), "tag": tag, "position": args.position,
            "scanned": len(targets), "changed": changed, "skipped": skipped,
            "dry_run": ctx["dry_run"], "results": results}


def cmd_drop(args, ctx):
    rate = args.rate
    if not (0.0 < rate <= 1.0):
        raise ToolError("--rate must be in the range (0, 1].")
    samples, _ = scan_dataset(ctx["data_dir"], strict=not ctx["allow_orphans"])
    targets = resolve_targets(samples, parse_tag_list(args.only), args.mode) if args.only else samples
    trash = Path(args.trash_dir).expanduser()
    if args.seed is not None:
        random.seed(args.seed)
    results = []
    for s in targets:
        if random.random() > rate:
            continue
        if not ctx["dry_run"]:
            if s["txt"]:
                _move_replace(Path(s["txt"]), trash)
            _move_replace(Path(s["image"]), trash)
        results.append({"name": s["name"], "image": s["image"], "txt": s["txt"]})
    return {"command": "drop", "data_dir": str(ctx["data_dir"]), "rate": rate, "trash_dir": str(trash),
            "scanned": len(targets), "dropped": len(results), "dry_run": ctx["dry_run"], "results": results}


def cmd_check(args, ctx):
    samples, orphan_txt = scan_dataset(ctx["data_dir"], strict=False)
    images_without_txt = [s["name"] for s in samples if not s["txt"]]
    ok = not orphan_txt
    return {"command": "check", "ok": ok, "data_dir": str(ctx["data_dir"]),
            "total_images": len(samples),
            "total_captions": len(samples) - len(images_without_txt),
            "orphan_txt": orphan_txt, "images_without_txt": images_without_txt}


HANDLERS = {
    "list": cmd_list,
    "show": cmd_show,
    "set": cmd_set,
    "stats": cmd_stats,
    "filter": cmd_filter,
    "remove-tags": cmd_remove_tags,
    "add-tag": cmd_add_tag,
    "drop": cmd_drop,
    "check": cmd_check,
}


# --------------------------------------------------------------------------
# Output rendering
# --------------------------------------------------------------------------

def emit(result, fmt):
    if fmt == "json":
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    cmd = result["command"]
    if cmd in ("list", "filter"):
        print(f"Dataset: {result['data_dir']}")
        print(f"Total: {result['total']}")
        for it in result["items"]:
            tags = ", ".join(it["tags"]) or "(empty)"
            missing = "" if it["txt"] else "  (no caption file)"
            print(f"{it['name']:<24}{tags}{missing}")
    elif cmd == "show":
        print(f"name: {result['name']}")
        print(f"image: {result['image']}")
        print(f"txt: {result['txt'] or '(none)'}")
        print(f"tags ({len(result['tags'])}): {', '.join(result['tags']) or '(empty)'}")
        print("caption:")
        print(result["caption"])
    elif cmd == "set":
        if result["dry_run"]:
            print("DRY RUN - no file written")
        action = "created" if result["created"] else "wrote"
        print(f"{action} caption ({len(result['caption'])} chars) to {result['txt']}")
    elif cmd == "stats":
        print(f"Dataset: {result['data_dir']}  (samples: {result['total_samples']})")
        print(f"{'Rank':<6}{'Tag':<30}{'Count':<8}{'Frequency'}")
        for i, st in enumerate(result["stats"], 1):
            print(f"{i:<6}{st['tag']:<30}{st['count']:<8}{st['frequency']:.2f}%")
    elif cmd in ("remove-tags", "add-tag"):
        if result["dry_run"]:
            print("DRY RUN - no files changed")
        line = f"scanned: {result['scanned']}, changed: {result['changed']}"
        if "skipped" in result:
            line += f", skipped (already had tag): {result['skipped']}"
        print(line)
        for r in result["results"]:
            if cmd == "remove-tags":
                print(f"  {r['name']}: removed [{', '.join(r['removed_tags'])}] -> {', '.join(r['new_tags'])}")
            else:
                print(f"  {r['name']}: new tags -> {', '.join(r['new_tags'])}")
    elif cmd == "drop":
        if result["dry_run"]:
            print("DRY RUN - nothing moved")
        print(f"dropped {result['dropped']} of {result['scanned']} scanned samples "
              f"(rate {result['rate']}) to {result['trash_dir']}")
        for r in result["results"]:
            print(f"  {r['name']}")
    elif cmd == "check":
        print(f"Dataset: {result['data_dir']}")
        print(f"images: {result['total_images']}, captions: {result['total_captions']}")
        if result["orphan_txt"]:
            print(f"ERROR: orphan caption files: {', '.join(result['orphan_txt'])}")
        else:
            print("OK: no orphan caption files")
        if result["images_without_txt"]:
            print(f"WARNING: images without caption: {', '.join(result['images_without_txt'])}")


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------

HELP_DESCRIPTION = """\
AxlRanko dataset management CLI for AI agents.

This script exposes the core functionality of the AxlRanko desktop app
(browsing, caption editing, tag statistics, and bulk dataset cleanup) as a
non-interactive command-line tool designed to be called by AI agents. It never
reads or decodes image pixels - it only works with the caption (.txt) files.

DATASET LAYOUT
  A training dataset directory contains image files (jpg, jpeg, png, webp, bmp)
  with a same-named .txt caption file next to each image. The caption content is
  a comma-separated list of tags, e.g. "1girl, solo, white hair".

DATA DIRECTORY (required, specified at startup)
  The dataset directory is never auto-discovered (the desktop app reads
  ../trainer/config.toml; this CLI does not). Provide it explicitly on every
  invocation with exactly one of:
    --data-dir DIR   path to the dataset directory
    --config PATH    path to a TOML file whose [environment].train_data_dir is
                     the dataset directory (e.g. the trainer/config.toml used
                     by the desktop app)
  The option may be given before or after the command.

COMMANDS
  list         List all samples in the dataset
  show         Show the caption of a single sample (text only, no image decoding)
  set          Write the caption of a single sample
  stats        Tag frequency statistics (each tag counted once per file)
  filter       Filter samples by tags (AND / OR)
  remove-tags  Remove tags from captions in bulk
  add-tag      Add a tag to captions in bulk (start or end, skips existing)
  drop         Randomly move samples (image + caption) to a trash directory
  check        Verify dataset integrity (orphan caption files, missing captions)

OUTPUT
  Results are printed to stdout as JSON (default) or human-readable text
  (--format text). Errors go to stderr. Exit codes:
    0  success
    1  runtime error (invalid dataset, not found, invalid value, ...)
    2  usage error (missing/invalid arguments)

SAFETY
  --dry-run  prints what a mutating command (set, remove-tags, add-tag, drop)
             would do without changing any file.
  stats/filter/remove-tags/add-tag/drop abort (exit 1) when the dataset contains
  an orphan caption file, mirroring the desktop app's integrity check. Use
  `check` to diagnose, or pass --allow-orphans to override. list/show are always
  lenient, like the desktop app's Images page.

EXAMPLES
  agent.py --data-dir ./dataset list
  agent.py list --data-dir ./dataset --limit 10
  agent.py --data-dir ./dataset show --name img001
  agent.py --data-dir ./dataset set --name img001 --caption "1girl, solo"
  agent.py --data-dir ./dataset stats --limit 20
  agent.py --data-dir ./dataset filter --tags "solo, 1girl" --mode and
  agent.py --data-dir ./dataset remove-tags --tags "blurry" --only "solo" --mode or --dry-run
  agent.py --config ../trainer/config.toml stats
  agent.py --data-dir ./dataset add-tag --tag "new" --position end
  agent.py --data-dir ./dataset drop --rate 0.2 --seed 42 --dry-run
  agent.py --data-dir ./dataset check
"""


def build_parser():
    # Common options are attached to the top-level parser AND every subparser,
    # so they can be given before or after the command, e.g. both
    # `agent.py --data-dir DIR list` and `agent.py list --data-dir DIR` work.
    # default=SUPPRESS keeps values set on the top-level parser from being
    # overwritten by subparser defaults (argparse merges the subparser's whole
    # namespace back into the parent one).
    common = argparse.ArgumentParser(add_help=False)
    src = common.add_mutually_exclusive_group()
    src.add_argument("--data-dir", metavar="DIR", default=argparse.SUPPRESS,
                     help="dataset directory (image files + same-named .txt captions)")
    src.add_argument("--config", metavar="PATH", default=argparse.SUPPRESS,
                     help="TOML config file whose [environment].train_data_dir is the dataset directory")
    common.add_argument("--format", choices=("json", "text"), default=argparse.SUPPRESS,
                        help="output format (default: json)")
    common.add_argument("--dry-run", action="store_true", default=argparse.SUPPRESS,
                        help="preview mutating commands without changing files")
    common.add_argument("--allow-orphans", action="store_true", default=argparse.SUPPRESS,
                        help="do not abort when orphan caption files are found (read-only inspection)")

    parser = argparse.ArgumentParser(
        prog="agent.py",
        description=HELP_DESCRIPTION,
        parents=[common],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    p = sub.add_parser("list", parents=[common], help="list all samples in the dataset",
                       description="List every sample (image + caption) in the dataset. "
                                   "Image pixels are never read.")
    p.add_argument("--limit", type=int, metavar="N", help="only show the first N samples (sorted by name)")

    p = sub.add_parser("show", parents=[common], help="show the caption of a single sample",
                       description="Show the caption text and parsed tags of one sample, "
                                   "matched by image or caption file stem (extension optional). "
                                   "Text only - image pixels are never read or decoded.")
    p.add_argument("--name", required=True, help="sample name, e.g. img001 or img001.png")

    p = sub.add_parser("set", parents=[common], help="write the caption of a single sample",
                       description="Write a caption to the sample's .txt file, creating it if it "
                                   "does not exist. The text is written exactly as given (the "
                                   "desktop app does not normalize single-sample saves).")
    p.add_argument("--name", required=True, help="sample name, e.g. img001 or img001.png")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--caption", metavar="TEXT", help="caption text to write")
    g.add_argument("--file", metavar="PATH", help="read caption text from a file")

    p = sub.add_parser("stats", parents=[common], help="tag frequency statistics",
                       description="Scan the dataset and report how many samples contain each tag "
                                   "and the percentage frequency. Each tag is counted once per file, "
                                   "mirroring the desktop app. Sorted by count (descending).")
    p.add_argument("--min-count", type=int, metavar="N", help="only show tags appearing in at least N samples")
    p.add_argument("--limit", type=int, metavar="N", help="only show the top N tags")

    p = sub.add_parser("filter", parents=[common], help="filter samples by tags (AND / OR)",
                       description="List samples whose tags match the given predicate: --mode and "
                                   "requires every tag, --mode or requires any tag (desktop app parity).")
    p.add_argument("--tags", required=True, metavar="TAGS",
                   help='comma-separated tags, e.g. "solo, 1girl"')
    p.add_argument("--mode", choices=("and", "or"), default="and",
                   help="match mode (default: and)")

    p = sub.add_parser("remove-tags", parents=[common], help="remove tags from captions in bulk",
                       description="Remove the given tags from the captions of matching samples "
                                   "(all samples, or only those matching --only). Only files that "
                                   "actually change are rewritten, in normalized ', ' format.")
    p.add_argument("--tags", required=True, metavar="TAGS",
                   help='comma-separated tags to remove, e.g. "blurry, low quality"')
    p.add_argument("--only", metavar="TAGS",
                   help="restrict to samples matching these tags (comma-separated)")
    p.add_argument("--mode", choices=("and", "or"), default="and",
                   help="match mode for --only (default: and)")

    p = sub.add_parser("add-tag", parents=[common], help="add a tag to captions in bulk",
                       description="Add one tag to the start or end of the captions of matching "
                                   "samples (all samples, or only those matching --only). Samples "
                                   "that already have the tag are skipped.")
    p.add_argument("--tag", required=True, metavar="TAG", help="tag to add")
    p.add_argument("--position", choices=("start", "end"), default="start",
                   help="where to insert the tag (default: start)")
    p.add_argument("--only", metavar="TAGS",
                   help="restrict to samples matching these tags (comma-separated)")
    p.add_argument("--mode", choices=("and", "or"), default="and",
                   help="match mode for --only (default: and)")

    p = sub.add_parser("drop", parents=[common], help="randomly move samples to a trash directory",
                       description="Move a random subset (probability --rate) of the matching "
                                   "samples - image and caption together - to --trash-dir, "
                                   "mirroring the desktop app's drop tool.")
    p.add_argument("--rate", type=float, required=True, metavar="R",
                   help="drop probability, 0 < R <= 1 (e.g. 0.2 drops 20 percent)")
    p.add_argument("--only", metavar="TAGS",
                   help="restrict to samples matching these tags (comma-separated)")
    p.add_argument("--mode", choices=("and", "or"), default="and",
                   help="match mode for --only (default: and)")
    p.add_argument("--trash-dir", default=DEFAULT_TRASH_DIR, metavar="DIR",
                   help=f"trash directory (default: {DEFAULT_TRASH_DIR})")
    p.add_argument("--seed", type=int, metavar="N", help="random seed for reproducible drops")

    p = sub.add_parser("check", parents=[common], help="verify dataset integrity",
                       description="Report orphan caption files (a .txt with no matching image) "
                                   "and images without a caption. Exits with code 1 when orphan "
                                   "caption files exist, mirroring the desktop app's strict scan.")
    return parser


def resolve_data_dir(args):
    if args.data_dir and args.config:
        raise ToolError("Provide only one of --data-dir or --config.")
    if args.data_dir:
        d = Path(args.data_dir).expanduser()
    elif args.config:
        d = read_train_data_dir(Path(args.config).expanduser())
    else:
        raise ToolError("Exactly one of --data-dir or --config is required.")
    d = d.resolve()
    if not d.is_dir():
        raise ToolError(f"Dataset directory does not exist or is not a directory: {d}")
    return d


def read_train_data_dir(config_path):
    try:
        import tomllib
    except ImportError:
        raise ToolError("--config requires Python 3.11+ (tomllib). Use --data-dir instead.")
    try:
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        raise ToolError(f"Config file not found: {config_path}")
    except tomllib.TOMLDecodeError as e:
        raise ToolError(f"Failed to parse config file {config_path}: {e}")
    env = data.get("environment") or {}
    train_dir = env.get("train_data_dir")
    if not train_dir:
        raise ToolError(f"Config file {config_path} has no [environment].train_data_dir.")
    return Path(str(train_dir)).expanduser()


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    # Common options may be absent from the namespace when default=SUPPRESS.
    args.data_dir = getattr(args, "data_dir", None)
    args.config = getattr(args, "config", None)
    args.format = getattr(args, "format", "json")
    args.dry_run = getattr(args, "dry_run", False)
    args.allow_orphans = getattr(args, "allow_orphans", False)

    if not args.data_dir and not args.config:
        parser.error("exactly one of --data-dir or --config is required "
                     "(provide it before or after the command)")

    try:
        data_dir = resolve_data_dir(args)
        ctx = {"data_dir": data_dir, "dry_run": args.dry_run, "allow_orphans": args.allow_orphans}
        result = HANDLERS[args.command](args, ctx)
        emit(result, args.format)
        if args.command == "check" and not result.get("ok", True):
            sys.exit(1)
    except ToolError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
