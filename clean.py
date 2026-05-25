import os
import shutil
from pathlib import Path

# Import the configuration from your project
from config import TrainConfig


def clean_project():
    print("=" * 50)
    print("      LoRA Training Directory Cleaner      ")
    print("=" * 50)

    # Initialize configuration
    try:
        cfg = TrainConfig()
    except Exception as e:
        print(f"[Error] Failed to load TrainConfig: {e}")
        return

    # Extract target paths and names from config
    output_dir = Path(cfg.output_dir)
    logging_dir = Path(cfg.logging_dir)
    output_name = cfg.output_name

    # Define specific targets based on config parameters
    samples_dir = output_dir / f"{output_name}_samples"
    tensorboard_log_dir = logging_dir / output_name

    print(f"Loaded configuration for project: '{output_name}'")
    print(f"Base Output Directory: {output_dir}")
    print(f"Base Logging Directory: {logging_dir}\n")

    # 1. Clean Generated Samples
    print("-" * 40)
    print("Step 1: Cleaning sample images...")
    if samples_dir.exists() and samples_dir.is_dir():
        try:
            shutil.rmtree(samples_dir)
            print(f"[Success] Removed samples directory: {samples_dir}")
        except Exception as e:
            print(f"[Warning] Could not remove samples directory: {e}")
    else:
        print(f"[Info] No samples directory found at: {samples_dir}")

    # 2. Clean TensorBoard Logs
    print("\n" + "-" * 40)
    print("Step 2: Cleaning TensorBoard logs...")
    if tensorboard_log_dir.exists() and tensorboard_log_dir.is_dir():
        try:
            shutil.rmtree(tensorboard_log_dir)
            print(f"[Success] Removed project logs: {tensorboard_log_dir}")
        except Exception as e:
            print(f"[Warning] Could not remove log directory: {e}")
    else:
        print(f"[Info] No log directory found at: {tensorboard_log_dir}")

    # 3. Optional: Clean Trained Weights (Safetensors)
    print("\n" + "-" * 40)
    print("Step 3: Checking for existing weights...")

    # Find directories matching output_name patterns
    weight_dirs = []
    if output_dir.exists():
        for path in output_dir.iterdir():
            if path.is_dir() and path.name.startswith(output_name):
                # Skip the samples directory as it was handled in step 1
                if path.name == f"{output_name}_samples":
                    continue
                weight_dirs.append(path)

    if weight_dirs:
        print(f"Found {len(weight_dirs)} checkpoint directory/directories:")
        for d in weight_dirs:
            print(f"  - {d.name}/")
        
        print("")  # Blank line for readability
        # Prompt user for confirmation
        confirmation = input("Do you want to delete these trained weight checkpoints? (y/N): ").strip().lower()
        
        if confirmation in ['y', 'yes']:
            print("Deleting weights...")
            for d in weight_dirs:
                try:
                    shutil.rmtree(d)
                    print(f"  [Deleted] {d.name}")
                except Exception as e:
                    print(f"  [Error] Failed to delete {d.name}: {e}")
            print("[Success] Selected weight checkpoints have been removed.")
        else:
            print("[Info] Skipped weights deletion. Safe-saving checkpoints.")
    else:
        print(f"[Info] No matching weight checkpoints found for '{output_name}' in {output_dir}")

    print("\n" + "=" * 50)
    print("Cleanup task completed.")
    print("=" * 50)


if __name__ == "__main__":
    clean_project()