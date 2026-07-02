import os
import sys
import argparse
import shutil

def parse_tags(tag_string):
    """Splits a comma-separated string into a list of cleaned tags."""
    if not tag_string:
        return []
    return [tag.strip() for tag in tag_string.split(",") if tag.strip()]

def should_move_to_trash(txt_path, positive_tags, negative_tags):
    """
    Determines if a file should be moved based on tag conditions.
    Returns True if:
    1. The annotation contains ANY negative tag.
    2. The annotation does NOT contain ALL positive tags (or is empty when positive tags are required).
    """
    # If the text file doesn't exist, treat it as empty content
    if not os.path.exists(txt_path):
        content = ""
    else:
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()

    # Check negative tags: if ANY negative tag is present, move to trash
    for nt in negative_tags:
        if nt in content:
            return True

    # Check positive tags: if ANY positive tag is missing, move to trash
    # (Meaning the file must contain ALL specified positive tags to stay)
    for pt in positive_tags:
        if pt not in content:
            return True

    return False

def main():
    parser = argparse.ArgumentParser(description="Filter images and annotations based on tags.")
    parser.add_argument("dir", type=str, help="Target directory containing images and txt files.")
    parser.add_argument("-n", "--negative", type=str, default="", help="Negative tags (comma-separated). Move if ANY match.")
    parser.add_argument("-p", "--positive", type=str, default="", help="Positive tags (comma-separated). Move if ANY miss.")
    
    args = parser.parse_args()
    
    target_dir = args.dir
    if not os.path.isdir(target_dir):
        print(f"Error: Directory '{target_dir}' does not exist.")
        sys.exit(1)

    negative_tags = parse_tags(args.negative)
    positive_tags = parse_tags(args.positive)

    # Create trash directory
    trash_dir = os.path.join(target_dir, "trash")
    os.makedirs(trash_dir, exist_ok=True)

    # Common image extensions to identify image files
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")

    # List all files in the directory
    all_files = os.listdir(target_dir)
    
    moved_count = 0

    for filename in all_files:
        file_path = os.path.join(target_dir, filename)
        
        # Skip directories (like the trash folder itself)
        if os.path.isdir(file_path):
            continue

        # Process only image files
        if filename.lower().endswith(image_extensions):
            base_name, _ = os.path.splitext(filename)
            txt_filename = base_name + ".txt"
            txt_path = os.path.join(target_dir, txt_filename)

            if should_move_to_trash(txt_path, positive_tags, negative_tags):
                # Move image
                shutil.move(file_path, os.path.join(trash_dir, filename))
                
                # Move annotation text file if it exists
                if os.path.exists(txt_path):
                    shutil.move(txt_path, os.path.join(trash_dir, txt_filename))
                
                moved_count += 1

    print(f"Filtering complete. Successfully moved {moved_count} image-text pairs to 'trash'.")

if __name__ == "__main__":
    main()
