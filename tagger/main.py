import os
import sys
import time
import glob
import onnxruntime as ort
import numpy as np
from PIL import Image
import pandas as pd

# Set environment variables for MIGraphX (AMD GPU) optimization
cache_dir = os.path.join(os.getcwd(), "migraphx_cache")
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

os.environ["ORT_MIGRAPHX_MODEL_CACHE_PATH"] = cache_dir
os.environ["ORT_MIGRAPHX_FP16_ENABLE"] = "1"
os.environ["ORT_MIGRAPHX_CACHE_PATH"] = cache_dir

# Enable readline for path auto-completion (Linux/macOS)
try:
    import readline

    def path_completer(text, state):
        target = os.path.expanduser(text)
        if not target:
            matches = glob.glob('*')
        else:
            matches = glob.glob(target + '*')
        matches = [m + os.sep if os.path.isdir(m) else m for m in matches]
        if state < len(matches):
            return matches[state]
        else:
            return None

    readline.set_completer_delims(' \t\n;')
    readline.parse_and_bind("tab: complete")
    readline.set_completer(path_completer)
except ImportError:
    print("[Warning] 'readline' module not available. Path auto-completion disabled.")


def load_labels(label_csv_path="selected_tags.csv"):
    """
    Loads labels and replaces underscores with spaces for standard tagging format.
    """
    df = pd.read_csv(label_csv_path)
    clean_tags = df['name'].astype(str).str.replace('_', ' ').tolist()
    return clean_tags


def process_image(image_path):
    """
    Preprocess image to match model requirements (448x448, RGB to BGR).
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((448, 448), resample=Image.Resampling.BICUBIC)
    image_data = np.array(img).astype(np.float32)
    # Convert RGB to BGR
    image_data = image_data[:, :, ::-1]
    image_data = np.expand_dims(image_data, axis=0)
    return image_data


def process_directory(directory_path, threshold, session, tag_names, input_name, output_name):
    """
    Processes all images in a directory and saves tags to .txt files.
    """
    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    image_files = [f for f in os.listdir(directory_path) if f.lower().endswith(valid_extensions)]

    if not image_files:
        print(f"Warning: No valid images found in '{directory_path}'.")
        return

    print(f"\nProcessing {len(image_files)} images with threshold: {threshold}")

    processed_count = 0
    start_time = time.time()

    for filename in image_files:
        image_path = os.path.join(directory_path, filename)
        txt_path = os.path.splitext(image_path)[0] + '.txt'

        try:
            image_tensor = process_image(image_path)
            confidences = session.run([output_name], {input_name: image_tensor})[0][0]

            # Filter tags based on user-provided threshold
            valid_tags = []
            for idx, score in enumerate(confidences):
                if score >= threshold:
                    valid_tags.append((tag_names[idx], score))

            # Sort by confidence (descending) - recommended for Danbooru-style tagging
            valid_tags.sort(key=lambda x: x[1], reverse=True)
            final_tags = [tag[0] for tag in valid_tags]
            tags_string = ", ".join(final_tags)

            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(tags_string)

            processed_count += 1
            print(f"[{processed_count}/{len(image_files)}] {filename} -> {len(final_tags)} tags.")

        except Exception as e:
            print(f"[Error] Failed to process {filename}: {e}")

    duration = time.time() - start_time
    print(f"\nCompleted! Processed {processed_count} images in {duration:.2f}s.")
    print("-" * 40 + "\n")


def run_inference():
    model_path = "model.onnx"
    label_path = "selected_tags.csv"

    providers = [
        ('MIGraphXExecutionProvider', {'device_id': 0}),
        'CPUExecutionProvider'
    ]

    try:
        print("Initializing session and warming up...")
        session = ort.InferenceSession(model_path, providers=providers)
        tag_names = load_labels(label_path)
    except Exception as e:
        print(f"Initialization Error: {e}")
        return

    active_provider = session.get_providers()[0]
    print(f"--- Session Ready ---")
    print(f"Active Provider: {active_provider}")
    print(f"Total Labels: {len(tag_names)}")
    print(f"----------------------\n")

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    while True:
        dir_input = input("Enter DIRECTORY path (or 'exit' to quit): ").strip()
        if dir_input.lower() in ['exit', 'quit', 'q']:
            break

        if not os.path.isdir(dir_input):
            print(f"Error: '{dir_input}' is not a valid directory.")
            continue

        # Threshold input and validation
        try:
            thresh_input = input("Enter confidence threshold [0.0 - 1.0] (default 0.35): ").strip()
            if thresh_input == "":
                threshold = 0.35
            else:
                threshold = float(thresh_input)
                if not (0.0 <= threshold <= 1.0):
                    print("Value out of range. Using default 0.35.")
                    threshold = 0.35
        except ValueError:
            print("Invalid input. Using default 0.35.")
            threshold = 0.35

        process_directory(dir_input, threshold, session, tag_names, input_name, output_name)


if __name__ == "__main__":
    run_inference()
