#!/usr/bin/env bash
# install.sh — expose start_ui.sh (the Chromatrix launcher) as a desktop application.
#
#   ./install.sh        install / refresh for the current user
#   ./install.sh -t     same, but keep a terminal window (build output stays visible)
#   ./install.sh -u     uninstall everything this script installed
#   ./install.sh -h     usage
#
# Artifacts (XDG dirs, defaulting under ~/.local):
#   $XDG_DATA_HOME/applications/axlranko.desktop
#   $XDG_DATA_HOME/icons/hicolor/<size>/apps/axlranko.png
#   $XDG_DATA_HOME/axlranko/launch.sh     console-hidden launcher (default mode only)
#   $XDG_STATE_HOME/axlranko/launch.log   written at launch time (default mode only)
#
# The Exec path is baked in as an absolute path, so re-run this script if the repo moves.
set -euo pipefail

APP_ID="axlranko"
APP_NAME="Chromatrix"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="$REPO_ROOT/start_ui.sh"
ICON_SRC="$REPO_ROOT/ranko/desktopApp/icons/app_icon.png"
JAR_DIR="$REPO_ROOT/ranko/desktopApp/build/compose/jars"

DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
STATE_HOME="${XDG_STATE_HOME:-$HOME/.local/state}"
APPS_DIR="$DATA_HOME/applications"
DESKTOP_FILE="$APPS_DIR/$APP_ID.desktop"
ICON_ROOT="$DATA_HOME/icons/hicolor"
WRAPPER_DIR="$DATA_HOME/$APP_ID"
WRAPPER="$WRAPPER_DIR/launch.sh"
LOG_FILE="$STATE_HOME/$APP_ID/launch.log"

# The first entry must be the full-size source icon; the rest are downscaled.
ICON_SIZES=(512 256 128 48)

usage() {
    cat <<'EOF'
Usage: install.sh [-t] [-u] [-h]

  (no option)  Install or refresh the Chromatrix desktop entry for this user.
  -t           Install a version that opens in a terminal, so the Gradle
               packaging output and startup errors stay visible.
  -u           Remove the desktop entry, the icon and the generated launcher.
  -h           Show this help.
EOF
}

die() {
    echo "install.sh: $*" >&2
    exit 1
}

# --- discovery of the toolchain the desktop session may not have on PATH -------

conda_base() {
    if [ -n "${CONDA_EXE:-}" ] && [ -x "${CONDA_EXE:-}" ]; then
        dirname "$(dirname "$CONDA_EXE")"
        return 0
    fi
    if command -v conda >/dev/null 2>&1; then
        conda info --base 2>/dev/null
        return 0
    fi
    local base
    for base in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" /opt/conda; do
        if [ -x "$base/bin/conda" ]; then
            echo "$base"
            return 0
        fi
    done
}

java_major_of() {
    local major
    major="$("$1" -version 2>&1 | head -n 1 | sed -E 's/.*version "([0-9]+).*/\1/')"
    case "$major" in
        1) echo 8 ;;
        [0-9]*) echo "$major" ;;
        *) echo 0 ;;
    esac
}

# Env/PATH lines for the generated launcher. A desktop session usually has a
# minimal PATH, so conda and the JDK are pinned at install time. Empty when
# nothing was detected — start_ui.sh has its own fallbacks.
build_env_block() {
    local conda_base_value java_bin major jdk path_prefix=""

    conda_base_value="$(conda_base || true)"
    if [ -n "$conda_base_value" ] && [ -x "$conda_base_value/bin/conda" ]; then
        printf 'export CONDA_EXE="%s"\n' "$conda_base_value/bin/conda"
        path_prefix="$conda_base_value/bin"
    fi

    if [ -n "${JAVA_HOME:-}" ] && [ -x "$JAVA_HOME/bin/java" ]; then
        java_bin="$JAVA_HOME/bin/java"
    else
        java_bin="$(command -v java || true)"
    fi
    if [ -n "$java_bin" ] && [ -x "$java_bin" ]; then
        major="$(java_major_of "$java_bin")"
        if [ "$major" -ge 21 ]; then
            jdk="$(dirname "$(dirname "$(readlink -f "$java_bin")")")"
            printf 'export AXL_JAVA="%s"\n' "$java_bin"
            printf 'export JAVA_HOME="%s"\n' "$jdk"
            path_prefix="${path_prefix:+$path_prefix:}$jdk/bin"
        else
            echo "install.sh: ignoring Java $major at $java_bin (start_ui.sh needs 21+)" >&2
        fi
    fi

    if [ -n "$path_prefix" ]; then
        printf 'export PATH="%s:$PATH"\n' "$path_prefix"
    fi
}

# --- icons --------------------------------------------------------------------

resize_icon() {
    local src="$1" dst="$2" size="$3"
    if command -v magick >/dev/null 2>&1; then
        magick "$src" -resize "${size}x${size}" "$dst"
    elif command -v convert >/dev/null 2>&1; then
        convert "$src" -resize "${size}x${size}" "$dst"
    elif command -v ffmpeg >/dev/null 2>&1; then
        ffmpeg -y -loglevel error -i "$src" -vf "scale=${size}:${size}" "$dst"
    else
        return 1
    fi
}

update_icon_cache() {
    if command -v gtk-update-icon-cache >/dev/null 2>&1; then
        gtk-update-icon-cache -q -t -f "$ICON_ROOT" >/dev/null 2>&1 || true
    fi
}

install_icons() {
    local size dst
    for size in "${ICON_SIZES[@]}"; do
        dst="$ICON_ROOT/${size}x${size}/apps/$APP_ID.png"
        mkdir -p "$(dirname "$dst")"
        if [ "$size" -eq "${ICON_SIZES[0]}" ]; then
            cp -f "$ICON_SRC" "$dst"
        else
            # No image tool available: keep the source size, the name is what matters.
            resize_icon "$ICON_SRC" "$dst" "$size" || cp -f "$ICON_SRC" "$dst"
        fi
    done
    update_icon_cache
}

remove_icons() {
    local found=no dst
    while IFS= read -r dst; do
        [ -f "$dst" ] || continue
        rm -f "$dst"
        rmdir "$(dirname "$dst")" 2>/dev/null || true
        rmdir "$(dirname "$(dirname "$dst")")" 2>/dev/null || true
        found=yes
    done < <(find "$ICON_ROOT" -mindepth 3 -maxdepth 3 -path "*/apps/$APP_ID.png" 2>/dev/null || true)
    [ "$found" = yes ] && update_icon_cache
    return 0
}

refresh_desktop_db() {
    if command -v update-desktop-database >/dev/null 2>&1; then
        update-desktop-database "$APPS_DIR" >/dev/null 2>&1 || true
    fi
}

# --- install ------------------------------------------------------------------

write_wrapper() {
    mkdir -p "$WRAPPER_DIR" "$(dirname "$LOG_FILE")"
    cat >"$WRAPPER" <<EOF
#!/usr/bin/env bash
# Generated by install.sh — desktop launcher for $APP_NAME. Re-run install.sh to refresh.
set -u

$(build_env_block)

LOG="$LOG_FILE"
mkdir -p "\$(dirname "\$LOG")"
echo "=== \$(date '+%F %T') launching from the desktop entry ===" >>"\$LOG"

if ! ls "$JAR_DIR"/*.jar >/dev/null 2>&1 && command -v notify-send >/dev/null 2>&1; then
    notify-send -a "$APP_NAME" "$APP_NAME is starting" \\
        "No packaged jar yet: the first build takes a few minutes." >/dev/null 2>&1 || true
fi

"$LAUNCHER" >>"\$LOG" 2>&1
code=\$?
if [ "\$code" -ne 0 ]; then
    echo "$APP_NAME exited with code \$code" >>"\$LOG"
    if command -v notify-send >/dev/null 2>&1; then
        notify-send -u critical -a "$APP_NAME" "$APP_NAME failed to start (exit \$code)" \\
            "See \$LOG" >/dev/null 2>&1 || true
    fi
fi
exit "\$code"
EOF
    chmod 755 "$WRAPPER"
}

write_desktop_file() {
    local terminal_mode="$1" exec_cmd
    if [ "$terminal_mode" = yes ]; then
        exec_cmd="\"$LAUNCHER\""
    else
        exec_cmd="\"$WRAPPER\""
    fi

    mkdir -p "$APPS_DIR"
    cat >"$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=$APP_NAME
GenericName=LoRA Trainer
Comment=Local-first LoRA training dashboard for Chromatrix
Exec=$exec_cmd
Icon=$APP_ID
Terminal=$( [ "$terminal_mode" = yes ] && echo true || echo false )
Categories=Graphics;2DGraphics;RasterGraphics;
Keywords=LoRA;SDXL;diffusion;training;Chromatrix;
StartupNotify=true
EOF
    chmod 755 "$DESKTOP_FILE"

    if command -v desktop-file-validate >/dev/null 2>&1; then
        desktop-file-validate "$DESKTOP_FILE" || true
    fi
}

do_install() {
    local terminal_mode="$1" icon_size="${ICON_SIZES[0]}"

    [ -f "$LAUNCHER" ] || die "start_ui.sh not found at $LAUNCHER"
    [ -x "$LAUNCHER" ] || die "start_ui.sh is not executable: chmod +x '$LAUNCHER'"
    [ -f "$ICON_SRC" ] || die "icon not found at $ICON_SRC"

    mkdir -p "$ICON_ROOT"
    if [ "$terminal_mode" = yes ]; then
        # Terminal mode runs start_ui.sh directly; drop a stale wrapper from a
        # previous default-mode install.
        rm -f "$WRAPPER"
        rmdir "$WRAPPER_DIR" 2>/dev/null || true
    else
        write_wrapper
    fi
    write_desktop_file "$terminal_mode"
    install_icons
    refresh_desktop_db

    echo "install.sh: installed $APP_NAME"
    echo "  desktop entry : $DESKTOP_FILE"
    echo "  icon          : $ICON_ROOT/${icon_size}x${icon_size}/apps/$APP_ID.png"
    if [ "$terminal_mode" = yes ]; then
        echo "  launcher      : $LAUNCHER  (Terminal=true, output visible)"
    else
        echo "  launcher      : $WRAPPER"
        echo "  launch log    : $LOG_FILE"
    fi
    echo "  uninstall     : $REPO_ROOT/install.sh -u"
}

# --- uninstall ----------------------------------------------------------------

do_uninstall() {
    local removed=no
    if [ -f "$DESKTOP_FILE" ]; then
        rm -f "$DESKTOP_FILE"
        removed=yes
    fi
    if [ -f "$WRAPPER" ]; then
        rm -f "$WRAPPER"
        rmdir "$WRAPPER_DIR" 2>/dev/null || true
        removed=yes
    fi
    if [ -f "$LOG_FILE" ]; then
        rm -f "$LOG_FILE"
        rmdir "$(dirname "$LOG_FILE")" 2>/dev/null || true
        removed=yes
    fi
    remove_icons
    refresh_desktop_db

    if [ "$removed" = yes ]; then
        echo "install.sh: removed the $APP_NAME desktop entry, icon and launcher"
    else
        echo "install.sh: nothing to remove ($APP_NAME is not installed)"
    fi
}

# --- main ---------------------------------------------------------------------

terminal_mode=no
action=install
for arg in "$@"; do
    case "$arg" in
        -u|--uninstall) action=uninstall ;;
        -t|--terminal) terminal_mode=yes ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option '$arg' (try -h)" ;;
    esac
done

if [ "$action" = uninstall ]; then
    do_uninstall
else
    do_install "$terminal_mode"
fi
