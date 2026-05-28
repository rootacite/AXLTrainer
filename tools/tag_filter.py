#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

def find_images_with_tags(directory_path, target_tags):
    dir_path = Path(directory_path)

    if not dir_path.exists() or not dir_path.is_dir():
        print(f"Error: The path '{directory_path}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    search_tags = {tag.strip().lower() for tag in target_tags.split(",") if tag.strip()}

    if not search_tags:
        print("Error: No valid target tags specified.", file=sys.stderr)
        sys.exit(1)

    matched_files_count = 0
    txt_files = list(dir_path.glob("*.txt"))

    print("=" * 60)
    print(" FILTER CRITERIA")
    print(f" Searching for images containing ALL of these tags:")
    for t in search_tags:
        print(f"  - {t}")
    print("=" * 60)
    print(f"{'Matching Image File':<45}{'Status':<15}")
    print("-" * 60)

    image_extensions = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]

    for txt_path in txt_files:
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                content = f.read()

            file_tags = {
                tag.strip().lower() for tag in content.replace('\n', ',').split(",") if tag.strip()
            }

            if search_tags.issubset(file_tags):
                corresponding_image = None
                for ext in image_extensions:
                    img_path = txt_path.with_suffix(ext)
                    if img_path.exists():
                        corresponding_image = img_path.name
                        break
                
                matched_files_count += 1
                status_str = "✓ Found" if corresponding_image else "✗ Missing Img"
                display_name = corresponding_image if corresponding_image else txt_path.stem + ".[?]"
                
                print(f"{display_name:<45}{status_str:<15}")

        except Exception as e:
            print(f"Warning: Failed to read {txt_path.name} due to {e}", file=sys.stderr)

    print("-" * 60)
    print(f" Total matching images found: {matched_files_count}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="List images that contain ALL specified tags in their corresponding .txt files."
    )
    parser.add_argument("dir", type=str, help="Path to the directory containing dataset files")
    parser.add_argument("tags", type=str, help="Comma-separated tags (e.g., '1boy, solo')")

    args = parser.parse_args()
    find_images_with_tags(args.dir, args.tags)