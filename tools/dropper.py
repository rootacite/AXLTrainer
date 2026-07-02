#!/usr/bin/env python3
import argparse
import random
import shutil
import sys
from pathlib import Path


def drop_samples(directory_path, tagger_name, rate):
    dir_path = Path(directory_path)

    if not dir_path.exists() or not dir_path.is_dir():
        print(f"Error: The path '{directory_path}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    if not (0.0 < rate <= 1.0):
        print("Error: Rate must be in the range (0, 1].", file=sys.stderr)
        sys.exit(1)

    trash_dir = dir_path / "trash"
    
    # Supported image extensions
    img_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    
    txt_files = list(dir_path.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in '{directory_path}'.")
        return

    processed_count = 0
    dropped_count = 0

    for txt_path in txt_files:
        # Skip files already inside trash if the script is rerun
        if txt_path.parent == trash_dir:
            continue

        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                content = f.read()

            tags = {tag.strip() for tag in content.split(",") if tag.strip()}
            
            if tagger_name in tags:
                processed_count += 1
                
                # Determine whether to drop based on the probability rate
                if random.random() < rate:
                    if not trash_dir.exists():
                        trash_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Find matching image file
                    img_path = None
                    for ext in img_extensions:
                        possible_img = txt_path.with_suffix(ext)
                        if possible_img.exists():
                            img_path = possible_img
                            break
                    
                    # Move txt file
                    shutil.move(str(txt_path), str(trash_dir / txt_path.name))
                    
                    # Move image file if it exists
                    if img_path:
                        shutil.move(str(img_path), str(trash_dir / img_path.name))
                        
                    dropped_count += 1

        except Exception as e:
            print(f"Warning: Failed to process {txt_path.name} due to {e}", file=sys.stderr)

    print("=" * 50)
    print(" DROPPER EXECUTION REPORT")
    print("=" * 50)
    print(f"Target Tag:       {tagger_name}")
    print(f"Drop Rate:        {rate * 100:.2f}%")
    print(f"Matching Files:   {processed_count}")
    print(f"Dropped Files:    {dropped_count}")
    print(f"Trash Directory:  {trash_dir}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Randomly drop dataset samples matching a specific tag into a trash folder."
    )
    parser.add_argument("dir", type=str, help="Path to the dataset directory")
    parser.add_argument("tag", type=str, help="The tag to target for dropping")
    parser.add_argument("rate", type=float, help="Probability of dropping (0.0, 1.0]")

    args = parser.parse_args()
    drop_samples(args.dir, args.tag, args.rate)
