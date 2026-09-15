#!/bin/bash
set -euo pipefail

# Launch the AxlRanko desktop UI with the trainer's conda env on PATH.
#
# Runs the packaged fat jar directly. Gradle is only invoked to repackage when
# the jar is missing or older than the Ranko sources, and it runs with
# --no-daemon so no Gradle/Kotlin daemon is left behind afterwards.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RANKO_DIR="$REPO_ROOT/ranko"
JAR_DIR="$RANKO_DIR/desktopApp/build/compose/jars"

# Inputs whose modification invalidates the packaged jar.
WATCHED=(
    "$RANKO_DIR/shared/src/commonMain"
    "$RANKO_DIR/shared/src/jvmMain"
    "$RANKO_DIR/desktopApp/src/main"
    "$RANKO_DIR/build.gradle.kts"
    "$RANKO_DIR/settings.gradle.kts"
    "$RANKO_DIR/gradle.properties"
    "$RANKO_DIR/gradle/libs.versions.toml"
    "$RANKO_DIR/gradle/wrapper/gradle-wrapper.properties"
    "$RANKO_DIR/shared/build.gradle.kts"
    "$RANKO_DIR/desktopApp/build.gradle.kts"
)

CONDA_SH="${CONDA_EXE:+$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh}"
if [ -z "${CONDA_SH:-}" ] || [ ! -f "$CONDA_SH" ]; then
    if command -v conda >/dev/null 2>&1; then
        CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
    else
        for base in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" /opt/conda; do
            if [ -f "$base/etc/profile.d/conda.sh" ]; then
                CONDA_SH="$base/etc/profile.d/conda.sh"
                break
            fi
        done
    fi
fi

if [ ! -f "${CONDA_SH:-}" ]; then
    echo "start_ui.sh: could not locate conda.sh; install conda or activate the 'axl' env manually." >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate axl_rocm_7_14

# Ranko spawns api.py itself (and api.py spawns the trainer). Point it at this
# env's interpreter instead of whatever `python3` resolves to on PATH.
export AXL_PYTHON="${CONDA_PREFIX}/bin/python"
export PYTHONUNBUFFERED=1

newest_jar() {
    [ -d "$JAR_DIR" ] || return 0
    ls -t "$JAR_DIR"/*.jar 2>/dev/null | head -n 1 || true
}

JAR="$(newest_jar)"

if [ -z "$JAR" ]; then
    echo "start_ui.sh: no packaged jar yet, building it..."
    (cd "$RANKO_DIR" && ./gradlew :desktopApp:packageUberJarForCurrentOS --no-daemon)
    JAR="$(newest_jar)"
elif [ -n "$(find "${WATCHED[@]}" -newer "$JAR" -print -quit 2>/dev/null || true)" ]; then
    echo "start_ui.sh: sources are newer than $(basename "$JAR"), repackaging..."
    (cd "$RANKO_DIR" && ./gradlew :desktopApp:packageUberJarForCurrentOS --no-daemon)
    JAR="$(newest_jar)"
fi

if [ -z "$JAR" ] || [ ! -f "$JAR" ]; then
    echo "start_ui.sh: packaging produced no jar under $JAR_DIR" >&2
    exit 1
fi

# JDK for the packaged jar: AXL_JAVA overrides JAVA_HOME, which overrides PATH.
# The jar is built for JDK 21 bytecode, so an older runtime on PATH will not do.
if [ -n "${AXL_JAVA:-}" ]; then
    JAVA_BIN="$AXL_JAVA"
elif [ -n "${JAVA_HOME:-}" ] && [ -x "$JAVA_HOME/bin/java" ]; then
    JAVA_BIN="$JAVA_HOME/bin/java"
else
    JAVA_BIN="$(command -v java || true)"
fi

if [ -z "$JAVA_BIN" ] || [ ! -x "$JAVA_BIN" ]; then
    echo "start_ui.sh: no java runtime found; set AXL_JAVA or JAVA_HOME to a JDK 21+ install." >&2
    exit 1
fi

# The app locates the repo by walking up from the working directory.
cd "$REPO_ROOT"
echo "start_ui.sh: launching $(basename "$JAR") with $JAVA_BIN"
exec "$JAVA_BIN" -jar "$JAR"
