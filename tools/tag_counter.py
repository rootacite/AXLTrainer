#!/usr/bin/env python3
import argparse
from collections import Counter
from pathlib import Path
import sys


def count_tags(directory_path):
    dir_path = Path(directory_path)

    # Check if the path exists and is a directory
    if not dir_path.exists():
        print(
            f"Error: The path '{directory_path}' does not exist.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not dir_path.is_dir():
        print(
            f"Error: The path '{directory_path}' is not a directory.",
            file=sys.stderr,
        )
        sys.exit(1)

    tag_counter = Counter()
    txt_file_count = 0

    # Find all *.txt files in the directory
    txt_files = list(dir_path.glob("*.txt"))

    if not txt_files:
        print(f"No .txt files found in '{directory_path}'.")
        return

    for file_path in txt_files:
        txt_file_count += 1
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

                # Split by comma and strip whitespaces/newlines
                # Using set() to ensure a tag is only counted ONCE per file
                tags = {tag.strip() for tag in content.split(",") if tag.strip()}

                # Update the global counter
                tag_counter.update(tags)
        except Exception as e:
            print(
                f"Warning: Failed to read {file_path.name} due to {e}",
                file=sys.stderr,
            )

    # Output the results
    print("=" * 50)
    print(f" TAG FREQUENCY REPORT (Total .txt files processed: {txt_file_count})")
    print("=" * 50)
    print(f"{'Rank':<6}{'Tag Name':<30}{'Count':<10}{'Frequency':<10}")
    print("-" * 50)

    # Sort by frequency (highest to lowest)
    for rank, (tag, count) in enumerate(tag_counter.most_common(), start=1):
        frequency = (count / txt_file_count) * 100
        print(f"{rank:<6}{tag:<30}{count:<10}{frequency:.2f}%")

    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count tag frequencies from .txt files in a directory (one count per file)."
    )
    parser.add_argument(
        "dir", type=str, help="Path to the directory containing .txt files"
    )

    args = parser.parse_args()
    count_tags(args.dir)