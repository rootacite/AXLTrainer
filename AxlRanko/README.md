# AxlRanko

AxlRanko is a desktop tool for managing AI model training datasets, built with Kotlin Multiplatform and Compose Multiplatform. It currently targets Desktop (JVM).

It focuses on the day-to-day maintenance of Stable Diffusion / LoRA training datasets: browsing and editing image captions, analyzing tag distribution, and cleaning up datasets in bulk based on tags.

## Features

### Image caption editing (Images)
- Browse the training dataset as a thumbnail list (jpg / jpeg / png / webp / bmp supported)
- Large preview plus a caption (tag) editor; unsaved edits are highlighted with a red border
- Save or reset the .txt caption file of the current image with one click
- Left/right and top/bottom pane ratios are adjustable by dragging the dividers

### Dataset statistics (Statistics)
- Scans the dataset and reports the occurrence count and percentage of every tag as a color-coded bar chart (higher frequency renders redder)
- Click tags to filter images, with both intersection (AND) and union (OR) logic modes
- Click any thumbnail in the result grid to jump to the caption editor, with that image preselected
- Dataset integrity check: aborts with an error if an orphan caption file (with no matching image) is found

### Bulk cleanup tools
- Remove selected tags: strips the selected tags from the captions of all matching images
- Add a tag in bulk: prepends or appends a tag to the captions of matching images (skips images that already have it)
- Drop samples: moves a random subset of matching images and their captions to `/tmp/axlranko/trash`, controlled by a probability rate r

### UI
- Material 3 design
- Draggable floating navigation rail; page transitions use a slide + fade animation

## Requirements and configuration

- JDK 17+ and Gradle (the project ships a wrapper)
- On startup the app locates its config automatically: it walks up from the executable's directory until it finds the `AxlRanko` directory, then reads `trainer/config.toml` in the sibling directory
- The config is a TOML file; `[environment].train_data_dir` points to the training dataset directory
- Dataset layout: image files and same-named `.txt` caption files stored side by side; caption content is a comma-separated list of tags

## Running

```bash
# Standard run (desktop app)
./gradlew :desktopApp:run

# Hot reload run
./gradlew :desktopApp:hotRun --auto

# Run tests
./gradlew :shared:jvmTest
```

## Project structure

- `desktopApp/` - desktop app entry point (Compose Desktop window)
- `shared/` - shared code (page UI, data models, config parsing)
  - `commonMain/` - common cross-platform code
  - `jvmMain/` - JVM-specific implementation (TOML parsing, etc.)

## Tech stack

- Kotlin Multiplatform / Compose Multiplatform (Desktop JVM)
- Material 3
- Metro (dependency injection) + metrox-viewmodel
- ktoml (TOML parsing) + kotlinx.serialization
- Coil 3 (image loading)
- okio / kotlinx.coroutines

## Note

- An example config lives in `trainer/config.toml` (sibling directory `trainer/` of this repository).
