
#!/usr/bin/env python3
"""
project_dump.py

Create a plain-text project dump with:
1. A tree-style project metadata section at the top.
2. Plain-text file content blocks for readable text files.
3. Output filename supplied from the command line.

Behavior:
- Hidden files/directories are skipped by default.
- A root-level .dumpignore file can exclude paths.
- Binary files are detected heuristically and skipped.
- Large files are truncated per --max-bytes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


DEFAULT_MAX_BYTES = 1 * 1024 * 1024  # 1 MiB
SAMPLE_CHUNK = 4096
OUTPUT_ENCODING = "utf-8"


@dataclass(slots=True)
class Entry:
    rel_path: str
    full_path: Path
    is_dir: bool
    is_binary: bool = False
    size: Optional[int] = None
    mtime: Optional[float] = None
    unreadable: bool = False
    ignored: bool = False


def format_mtime(epoch: Optional[float]) -> str:
    if epoch is None:
        return "N/A"
    return dt.datetime.fromtimestamp(epoch).isoformat(sep=" ", timespec="seconds")


def format_size(size: Optional[int]) -> str:
    if size is None:
        return "-"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(size)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= 1024.0
    return f"{size} B"


def is_binary_file(path: Path, sample_bytes: int = SAMPLE_CHUNK) -> bool:
    try:
        with path.open("rb") as f:
            chunk = f.read(sample_bytes)
    except Exception:
        return True

    if not chunk:
        return False

    if b"\x00" in chunk:
        return True

    try:
        chunk.decode("utf-8")
        return False
    except UnicodeDecodeError:
        return True


def normalize_ignore_path(raw: str) -> Optional[str]:
    line = raw.split("#", 1)[0].strip()
    if not line:
        return None
    if line.startswith("./") or line.startswith(".\\"):
        line = line[2:]
    norm = os.path.normpath(line)
    if norm in ("", "."):
        return None
    return norm


def read_dumpignore(root_path: Path) -> set[str]:
    ignore_file = root_path / ".dumpignore"
    if not ignore_file.is_file():
        return set()

    ignores: set[str] = set()
    try:
        for raw in ignore_file.read_text(encoding=OUTPUT_ENCODING).splitlines():
            item = normalize_ignore_path(raw)
            if item:
                ignores.add(item)
    except Exception:
        return set()
    return ignores


def is_ignored_path(norm_rel: str, ignore_paths: set[str]) -> bool:
    for ip in ignore_paths:
        if norm_rel == ip or norm_rel.startswith(ip + os.sep):
            return True
    return False


def should_skip_hidden(rel_parts: tuple[str, ...], name: str, include_hidden: bool) -> bool:
    if include_hidden:
        return False
    return name.startswith(".") or any(part.startswith(".") for part in rel_parts)


def gather_entries(
    root_path: str,
    follow_symlinks: bool = False,
    include_hidden: bool = False,
    ignore_paths: Optional[set[str]] = None,
) -> list[Entry]:
    root = Path(root_path).resolve()
    ignore_paths = ignore_paths or set()
    entries: list[Entry] = []
    seen_dirs: set[str] = set()

    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        dirnames.sort()
        filenames.sort()

        current_dir = Path(dirpath)
        rel_dir = os.path.relpath(current_dir, root)
        rel_dir_norm = "." if rel_dir == "." else os.path.normpath(rel_dir)
        rel_parts = () if rel_dir_norm in ("", ".") else tuple(Path(rel_dir_norm).parts)

        # Prevent descent into hidden directories unless explicitly enabled.
        filtered_dirnames: list[str] = []
        for d in dirnames:
            if should_skip_hidden(rel_parts, d, include_hidden):
                continue
            filtered_dirnames.append(d)
        dirnames[:] = filtered_dirnames

        # Record visible directories so they can appear in the tree section.
        if rel_dir_norm not in ("", ".") and rel_dir_norm not in seen_dirs:
            seen_dirs.add(rel_dir_norm)
            try:
                stat = current_dir.stat()
                entries.append(
                    Entry(
                        rel_path=rel_dir_norm,
                        full_path=current_dir,
                        is_dir=True,
                        size=stat.st_size,
                        mtime=stat.st_mtime,
                    )
                )
            except Exception:
                entries.append(
                    Entry(
                        rel_path=rel_dir_norm,
                        full_path=current_dir,
                        is_dir=True,
                        unreadable=True,
                    )
                )

        for filename in filenames:
            if should_skip_hidden(rel_parts, filename, include_hidden):
                continue

            full_path = current_dir / filename
            rel = os.path.relpath(full_path, root)
            norm_rel = os.path.normpath(rel)
            ignored = is_ignored_path(norm_rel, ignore_paths)

            try:
                stat = os.stat(full_path, follow_symlinks=follow_symlinks)
            except Exception:
                entries.append(
                    Entry(
                        rel_path=norm_rel,
                        full_path=full_path,
                        is_dir=False,
                        unreadable=True,
                    )
                )
                continue

            is_bin = False if ignored else is_binary_file(full_path)
            entries.append(
                Entry(
                    rel_path=norm_rel,
                    full_path=full_path,
                    is_dir=False,
                    is_binary=is_bin,
                    size=stat.st_size,
                    mtime=stat.st_mtime,
                    ignored=ignored,
                )
            )

    # Deterministic ordering: directories first, then files, both lexicographically.
    entries.sort(key=lambda e: (Path(e.rel_path).parts, 0 if e.is_dir else 1))
    return entries


def build_tree_lines(entries: list[Entry]) -> list[str]:
    """
    Build a tree view rooted at '.'.
    Example:
    .
    ├── src/
    │   └── main.py [text, 1.2 KiB]
    └── README.md [text, 2.0 KiB]
    """
    dir_set: set[str] = set()
    file_map: dict[str, Entry] = {}

    def add_dir(path: str) -> None:
        if path in ("", "."):
            return
        dir_set.add(path)
        parts = list(Path(path).parts)
        acc: list[str] = []
        for part in parts[:-1]:
            acc.append(part)
            dir_set.add(os.path.normpath(os.path.join(*acc)))

    for e in entries:
        parts = list(Path(e.rel_path).parts)
        if e.is_dir:
            add_dir(e.rel_path)
        else:
            file_map[e.rel_path] = e
            for i in range(1, len(parts)):
                dir_set.add(os.path.normpath(os.path.join(*parts[:i])))

    children: dict[str, set[str]] = {"": set()}
    for d in dir_set:
        children.setdefault(d, set())
        parent = str(Path(d).parent)
        if parent == ".":
            parent = ""
        children.setdefault(parent, set()).add(d)

    for path in file_map:
        parent = str(Path(path).parent)
        if parent == ".":
            parent = ""
        children.setdefault(parent, set()).add(path)

    def is_dir_node(path: str) -> bool:
        return path in dir_set and path not in file_map

    def label(path: str) -> str:
        if path == ".":
            return "."
        if is_dir_node(path):
            return f"{Path(path).name}/"
        e = file_map[path]
        flags: list[str] = []
        if e.unreadable:
            flags.append("unreadable")
        elif e.ignored:
            flags.append("ignored")
        elif e.is_binary:
            flags.append("binary")
        else:
            flags.append("text")
        if e.size is not None:
            flags.append(format_size(e.size))
        return f"{Path(path).name} [{', '.join(flags)}]"

    def sort_key(path: str) -> tuple[int, str]:
        return (0 if is_dir_node(path) else 1, Path(path).name.lower())

    lines: list[str] = ["."]

    def render(parent: str, prefix: str = "") -> None:
        items = sorted(children.get(parent, set()), key=sort_key)
        for idx, child in enumerate(items):
            last = idx == len(items) - 1
            connector = "└── " if last else "├── "
            next_prefix = "    " if last else "│   "
            lines.append(prefix + connector + label(child))
            if is_dir_node(child):
                render(child, prefix + next_prefix)

    render("")
    return lines


def read_text_file(path: Path, max_bytes: int) -> tuple[str, bool]:
    with path.open("rb") as f:
        raw = f.read(max_bytes + 1)
    truncated = len(raw) > max_bytes
    if truncated:
        raw = raw[:max_bytes]
    return raw.decode("utf-8", errors="replace"), truncated


def build_output(
    entries: list[Entry],
    root_path: Path,
    output_file: Path,
    max_bytes: int,
    verbose: bool,
) -> str:
    root_path = root_path.resolve()
    generated = dt.datetime.now().isoformat(sep=" ", timespec="seconds")

    file_entries = [e for e in entries if not e.is_dir]
    included = 0
    skipped_binary = 0
    skipped_ignored = 0
    unreadable = 0
    truncated = 0

    tree_lines = build_tree_lines(entries)
    blocks: list[str] = []

    for e in file_entries:
        rel = e.rel_path
        mtime = format_mtime(e.mtime)

        if e.unreadable:
            unreadable += 1
            continue
        if e.ignored:
            skipped_ignored += 1
            continue
        if e.is_binary:
            skipped_binary += 1
            continue

        try:
            text, was_truncated = read_text_file(e.full_path, max_bytes)
        except Exception:
            unreadable += 1
            continue

        included += 1
        if was_truncated:
            truncated += 1

        status = "included"
        if was_truncated:
            status = f"included (truncated to {max_bytes} bytes)"

        blocks.append(
            "\n".join(
                [
                    f"===== FILE: {rel} =====",
                    f"path: {rel}",
                    f"size: {e.size if e.size is not None else '-'}",
                    f"modified: {mtime}",
                    f"status: {status}",
                    "",
                    "----- BEGIN TEXT -----",
                    text,
                    "----- END TEXT -----",
                    "",
                ]
            )
        )

    summary = "\n".join(
        [
            "SUMMARY",
            f"root: {root_path}",
            f"generated: {generated}",
            f"total entries: {len(entries)}",
            f"text files included: {included}",
            f"binary files skipped: {skipped_binary}",
            f"ignored files skipped: {skipped_ignored}",
            f"unreadable files: {unreadable}",
            f"truncated files: {truncated}",
            "",
        ]
    )

    output_text = "\n".join(
        [
            "PROJECT METADATA",
            f"root: {root_path}",
            f"generated: {generated}",
            f"output: {output_file}",
            "",
            "TREE",
            *tree_lines,
            "",
            summary,
            "FILE CONTENTS",
            "",
            *blocks,
        ]
    )

    if verbose:
        print(f"Writing dump to: {output_file}")
        print(
            "Files included: "
            f"{included}, binary skipped: {skipped_binary}, ignored skipped: {skipped_ignored}, "
            f"unreadable: {unreadable}, truncated: {truncated}"
        )

    return output_text


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a plain-text dump of a directory tree with metadata and file contents."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Root directory to dump (default: current directory).",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output dump file name or path.",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=DEFAULT_MAX_BYTES,
        help=f"Maximum bytes to include per text file (default: {DEFAULT_MAX_BYTES}).",
    )
    parser.add_argument(
        "--follow-symlinks",
        action="store_true",
        help="Follow symbolic links while walking the tree.",
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Include hidden files and directories.",
    )
    parser.add_argument(
        "--no-verbose",
        dest="verbose",
        action="store_false",
        help="Suppress console output.",
    )
    parser.set_defaults(verbose=True)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    root = Path(args.root)
    output = Path(args.output)

    ignore_paths = read_dumpignore(root)
    entries = gather_entries(
        str(root),
        follow_symlinks=args.follow_symlinks,
        include_hidden=args.include_hidden,
        ignore_paths=ignore_paths,
    )

    text = build_output(
        entries=entries,
        root_path=root,
        output_file=output,
        max_bytes=args.max_bytes,
        verbose=args.verbose,
    )

    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding=OUTPUT_ENCODING)
    except Exception as e:
        raise RuntimeError(f"Failed to write output file '{output}': {e}") from e

    return 0


if __name__ == "__main__":
    sys.exit(main())