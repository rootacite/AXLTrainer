import os
import sys
import random
import uuid

def shuffle_dataset(target_dir):
    if not os.path.isdir(target_dir):
        print(f"Error: {target_dir} is not a valid directory.")
        return

    # Group files by their base name (filename without extension)
    files = os.listdir(target_dir)
    file_groups = {}
    
    for f in files:
        base, ext = os.path.splitext(f)
        if base not in file_groups:
            file_groups[base] = []
        file_groups[base].append(ext)

    # Get all unique base names and shuffle them
    base_names = list(file_groups.keys())
    random.shuffle(base_names)
    
    # Use a mapping to ensure unique new names and avoid overwriting existing files
    # Step 1: Rename to temporary unique UUIDs to avoid collisions
    temp_mapping = []
    for base in base_names:
        temp_name = str(uuid.uuid4())
        for ext in file_groups[base]:
            old_path = os.path.join(target_dir, base + ext)
            new_path = os.path.join(target_dir, temp_name + ext)
            os.rename(old_path, new_path)
        temp_mapping.append((temp_name, file_groups[base]))

    # Step 2: Rename from UUIDs to final shuffled numeric sequence
    # Using 4-digit padding (0001, 0002, etc.) for better sorting
    for index, (temp_name, extensions) in enumerate(temp_mapping, start=1):
        final_base = f"{index:04d}"
        for ext in extensions:
            old_path = os.path.join(target_dir, temp_name + ext)
            new_path = os.path.join(target_dir, final_base + ext)
            os.rename(old_path, new_path)

    print(f"Successfully shuffled {len(base_names)} file groups in '{target_dir}'.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python shuffle.py <directory_path>")
    else:
        shuffle_dataset(sys.argv[1])
