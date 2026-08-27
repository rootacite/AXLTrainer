from pathlib import Path

from trainer.cleanup import discover, run_cleanup
from trainer.config import TrainConfig


def clean_project() -> None:
    print("=" * 50)
    print("      LoRA Training Directory Cleaner      ")
    print("=" * 50)

    try:
        cfg = TrainConfig()
    except Exception as e:
        print(f"[Error] Failed to load TrainConfig: {e}")
        return

    output_dir = Path(cfg.output_dir)
    logging_dir = Path(cfg.logging_dir)
    output_name = cfg.output_name
    plan = discover(output_dir, logging_dir, output_name)

    print(f"Loaded configuration for project: '{output_name}'")
    print(f"Base Output Directory: {output_dir}")
    print(f"Base Logging Directory: {logging_dir}\n")

    print("-" * 40)
    print("Step 1: Cleaning sample images...")
    samples_dir = plan["samples_dir"]
    if samples_dir.exists() and samples_dir.is_dir():
        print(f"[Info] Will remove samples directory: {samples_dir}")
    else:
        print(f"[Info] No samples directory found at: {samples_dir}")

    print("\n" + "-" * 40)
    print("Step 2: Cleaning TensorBoard logs...")
    log_dir = plan["log_dir"]
    if log_dir.exists() and log_dir.is_dir():
        print(f"[Info] Will remove project logs: {log_dir}")
    else:
        print(f"[Info] No log directory found at: {log_dir}")

    print("\n" + "-" * 40)
    print("Step 3: Checking for existing weights...")
    weight_dirs = plan["weight_dirs"]
    delete_weights = False
    if weight_dirs:
        print(f"Found {len(weight_dirs)} checkpoint directory/directories:")
        for directory in weight_dirs:
            print(f"  - {directory.name}/")
        print("")
        confirmation = input("Do you want to delete these trained weight checkpoints? (y/N): ").strip().lower()
        delete_weights = confirmation in ("y", "yes")
        if not delete_weights:
            print("[Info] Skipped weights deletion. Safe-saving checkpoints.")
    else:
        print(f"[Info] No matching weight checkpoints found for '{output_name}' in {output_dir}")

    result = run_cleanup(output_dir, logging_dir, output_name, delete_weights=delete_weights)
    for path in result["removed"]:
        print(f"[Deleted] {path}")
    for message in result["errors"]:
        print(f"[Error] {message}")

    print("\n" + "=" * 50)
    print("Cleanup task completed.")
    print("=" * 50)


if __name__ == "__main__":
    clean_project()
