import os
import sys
import random
from PIL import Image

def process_images(stand_dir, bg_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    stand_files = [f for f in os.listdir(stand_dir) if f.lower().endswith('.png')]
    bg_files = [f for f in os.listdir(bg_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not stand_files:
        print("Error: No PNG images found in stand directory.")
        return
    if not bg_files:
        print("Error: No images found in background directory.")
        return

    for stand_name in stand_files:
        stand_path = os.path.join(stand_dir, stand_name)
        bg_name = random.choice(bg_files)
        bg_path = os.path.join(bg_dir, bg_name)

        try:
            with Image.open(stand_path) as stand_img, Image.open(bg_path) as bg_img:
                stand_img = stand_img.convert("RGBA")
                bg_img = bg_img.convert("RGBA")

                bg_w, bg_h = bg_img.size
                st_w, st_h = stand_img.size

                new_h = bg_h
                new_w = int(st_w * (bg_h / st_h))
                stand_resized = stand_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

                composed_layer = Image.new("RGBA", (bg_w, bg_h), (0, 0, 0, 0))

                paste_x = (bg_w - new_w) // 2
                paste_y = 0

                composed_layer.paste(stand_resized, (paste_x, paste_y))

                final_img = Image.alpha_composite(bg_img, composed_layer)

                output_path = os.path.join(output_dir, stand_name)
                final_img.save(output_path, "PNG")
                print(f"Composed: {stand_name} with {bg_name} -> {output_path}")

        except Exception as e:
            print(f"Failed to process {stand_name}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python rc.py <stand_dir> <bg_dir> <output_dir>")
        sys.exit(1)

    process_images(sys.argv[1], sys.argv[2], sys.argv[3])
