#!/usr/bin/env bash
# Wanalizer - launcher
# --------------------------------------------------------------------------
# Auto-installs everything needed and picks the best Python interpreter:
#
#   .venv      - system Python (with GIL) + PySide6 + every GUI dep
#                Used for the GUI process and as the default CLI runtime.
#
#   .venv-t    - free-threaded Python 3.14t (no-GIL) used by `--ft` mode
#                for true parallel threading. Falls back to .venv if not
#                available.
#
# CPU-heavy code uses concurrent.futures.ProcessPoolExecutor so it always
# scales regardless of GIL. I/O (Ollama calls, file hashing) uses threads.
# --------------------------------------------------------------------------
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

VENV_DIR="$PROJECT_DIR/.venv"
VENV_T_DIR="$PROJECT_DIR/.venv-t"
PYTHON_T_DIR="$PROJECT_DIR/.python-t"

# ---- Constants ----
FREE_THREADED_PY_VERSION="3.14.6"
FREE_THREADED_PY_DATE="20260610"
FREE_THREADED_PY_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${FREE_THREADED_PY_DATE}/cpython-${FREE_THREADED_PY_VERSION}+${FREE_THREADED_PY_DATE}-x86_64-unknown-linux-gnu-freethreaded-install_only.tar.gz"

CORE_DEPS=( "Pillow>=11" "numpy>=2.1" "requests>=2.28" )
GUI_DEPS=( "PySide6>=6.6" )
OPTIONAL_DEPS=( "imagehash>=4.3" "scikit-learn>=1.3" )

# ---- Detect Qt platform ----
detect_qt_platform() {
    if [ -n "${WAYLAND_DISPLAY:-}" ]; then
        echo "wayland"
    elif [ -n "${DISPLAY:-}" ]; then
        echo "xcb"
    else
        echo "offscreen"
    fi
}
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-$(detect_qt_platform)}"

# ---- Helpers ----
say()  { printf '\033[1;34m[run.sh]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[run.sh]\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[1;31m[run.sh]\033[0m %s\n' "$*" >&2; }
have() { command -v "$1" >/dev/null 2>&1; }
is_macos() { [ "$(uname -s)" = "Darwin" ]; }
is_windows() { case "$(uname -s 2>/dev/null || echo Windows)" in MINGW*|MSYS*|CYGWIN*|Windows*) return 0;; *) return 1;; esac; }

system_python() {
    # Pick the newest system python3
    for c in python3.14 python3.13 python3.12 python3; do
        if have "$c"; then
            echo "$c"
            return
        fi
    done
}

venv_python() {
    [ -f "$1/bin/python" ] && echo "$1/bin/python" || echo ""
}

have_mod() {
    "$1" -c "import $2" >/dev/null 2>&1
}

# ---- Bootstrap system venv (.venv) ----
bootstrap_system_venv() {
    local py
    py="$(venv_python "$VENV_DIR")"
    if [ -z "$py" ]; then
        local sys_py
        sys_py="$(system_python)"
        if [ -z "$sys_py" ]; then
            err "No system python3 found. Install Python 3.10+ first."
            exit 1
        fi
        say "Creating system venv at $VENV_DIR using $sys_py ..."
        "$sys_py" -m venv "$VENV_DIR"
        py="$VENV_DIR/bin/python"
    fi

    say "Upgrading pip in $VENV_DIR ..."
    "$py" -m pip install --quiet --upgrade pip wheel setuptools

    local missing=()
    for mod in "${CORE_DEPS[@]}"; do
        local pkg="${mod%%>=*}"; local pkg="${pkg%%<*}"; local pkg="${pkg%%=*}"; local pkg="${pkg%%~*}"
        have_mod "$py" "$pkg" || missing+=("$mod")
    done
    if [ "${#missing[@]}" -gt 0 ]; then
        say "Installing core deps: ${missing[*]}"
        "$py" -m pip install --quiet "${missing[@]}"
    fi

    # GUI deps (best effort - PySide6 only on x86_64 Linux/macOS/Windows)
    local gui_missing=()
    for mod in "${GUI_DEPS[@]}"; do
        local pkg="${mod%%>=*}"; local pkg="${pkg%%<*}"; local pkg="${pkg%%=*}"; local pkg="${pkg%%~*}"
        have_mod "$py" "$pkg" || gui_missing+=("$mod")
    done
    if [ "${#gui_missing[@]}" -gt 0 ]; then
        say "Installing GUI deps: ${gui_missing[*]}"
        "$py" -m pip install --quiet "${gui_missing[@]}" 2>/dev/null \
            && say "GUI deps installed." \
            || warn "GUI deps install failed (you will not be able to launch the GUI)."
    fi
}

# ---- Bootstrap free-threaded venv (.venv-t) ----
bootstrap_ft_venv() {
    local py_t
    py_t="$(venv_python "$VENV_T_DIR")"
    if [ -z "$py_t" ]; then
        if is_macos || is_windows; then
            warn "Free-threaded Python bootstrap is Linux x86_64 only; skipping."
            return 1
        fi
        say "Downloading free-threaded Python ${FREE_THREADED_PY_VERSION} ..."
        if [ ! -x "$PYTHON_T_DIR/bin/python3" ]; then
            mkdir -p "$PYTHON_T_DIR"
            local tarball="/tmp/cpython-ft-${FREE_THREADED_PY_VERSION}.tar.gz"
            if [ ! -f "$tarball" ]; then
                say "  -> ${FREE_THREADED_PY_URL}"
                if ! curl -fL --connect-timeout 15 --max-time 120 \
                       "$FREE_THREADED_PY_URL" -o "$tarball" 2>/dev/null; then
                    warn "Could not download free-threaded Python (network error); skipping."
                    rm -f "$tarball"
                    return 1
                fi
                if [ ! -s "$tarball" ]; then
                    warn "Downloaded file is empty; skipping."
                    rm -f "$tarball"
                    return 1
                fi
            fi
            say "Extracting to $PYTHON_T_DIR ..."
            if ! tar -xzf "$tarball" -C "$PYTHON_T_DIR" --strip-components=1 2>/dev/null; then
                warn "Extraction failed; skipping."
                rm -f "$tarball"
                rm -rf "$PYTHON_T_DIR"
                return 1
            fi
            rm -f "$tarball"
        fi
        if [ ! -x "$PYTHON_T_DIR/bin/python3" ]; then
            warn "Free-threaded Python not available after extraction."
            return 1
        fi
        say "Creating free-threaded venv at $VENV_T_DIR ..."
        if ! "$PYTHON_T_DIR/bin/python3" -m venv "$VENV_T_DIR" 2>/dev/null; then
            warn "Failed to create free-threaded venv; skipping."
            return 1
        fi
        py_t="$VENV_T_DIR/bin/python"
    fi

    say "Upgrading pip in $VENV_T_DIR ..."
    "$py_t" -m pip install --quiet --upgrade pip wheel setuptools

    local missing=()
    for mod in "${CORE_DEPS[@]}" "${OPTIONAL_DEPS[@]}"; do
        local pkg="${mod%%>=*}"; local pkg="${pkg%%<*}"; local pkg="${pkg%%=*}"; local pkg="${pkg%%~*}"
        have_mod "$py_t" "$pkg" || missing+=("$mod")
    done
    if [ "${#missing[@]}" -gt 0 ]; then
        say "Installing free-threaded deps: ${missing[*]}"
        "$py_t" -m pip install --quiet "${missing[@]}" 2>/dev/null \
            || warn "Some free-threaded deps failed (they may not yet have no-GIL wheels)."
    fi

    # Verify it really is free-threaded
    if "$py_t" -c "import sys; sys.flags.gil" 2>/dev/null | grep -q None; then
        say "Free-threaded Python ready ($($py_t --version))."
    else
        warn "$VENV_T_DIR is not free-threaded (sys.flags.gil is set)."
    fi
}

# ---- Bootstrap (all) ----
bootstrap() {
    bootstrap_system_venv
    bootstrap_ft_venv || true
    say "Optional deps: opencv-python-headless (skipped, no free-thread wheel yet)"
    say "Done."
}

# ---- Pick runtime ----
pick_python() {
    local want_ft="${1:-no}"
    if [ "$want_ft" = "yes" ] && [ -x "$VENV_T_DIR/bin/python" ]; then
        echo "$VENV_T_DIR/bin/python"
    else
        echo "$VENV_DIR/bin/python"
    fi
}

# ---- Dispatch ----
case "${1:-}" in
    --cli|-c)
        shift
        want_ft="no"
        if [ "${1:-}" = "--ft" ]; then
            want_ft="yes"
            shift
        fi
        py="$(pick_python "$want_ft")"
        say "Using $( [ "$want_ft" = "yes" ] && echo "free-threaded" || echo "system" ) Python: $py"
        if [ ! -f "$py" ]; then
            bootstrap
            py="$(pick_python "$want_ft")"
        fi
        exec "$py" -m wallpaper_analyzer.cli "$@"
        ;;

    --bootstrap)
        bootstrap
        exit 0
        ;;

    --bootstrap-ft)
        bootstrap_ft_venv
        exit 0
        ;;

    --version|-V)
        py="$(venv_python "$VENV_DIR")"
        if [ -z "$py" ]; then
            system_python >/dev/null && py="$(system_python)" || py="python3"
        fi
        say "Wanalizer v$(.venv/bin/python -c 'import wallpaper_analyzer; print(wallpaper_analyzer.__version__)' 2>/dev/null || echo 'unknown')"
        say "Launcher: $0"
        say "Project dir: $PROJECT_DIR"
        exit 0
        ;;

    --ft)
        # Force free-threaded Python even for GUI (will fall back to CLI if no PySide6)
        shift
        py="$(pick_python yes)"
        if [ ! -f "$py" ]; then
            bootstrap_ft_venv || bootstrap_system_venv
            py="$(pick_python yes)"
        fi
        if have_mod "$py" PySide6; then
            exec "$py" -m wallpaper_analyzer.gui "$@"
        else
            warn "PySide6 not available in free-threaded venv."
            warn "Falling back to CLI mode (use ./run.sh --cli for the same)."
            exec "$py" -m wallpaper_analyzer.cli "$@"
        fi
        ;;

    --help|-h)
        cat <<EOF
Wanalizer launcher

Usage: $0 [OPTIONS]

Startup options:
  (no args)               Launch GUI (uses system Python + PySide6)
  --cli, -c [--ft]        Run in CLI mode; --ft picks the free-threaded Python
  --ft                    Force free-threaded Python (falls back to CLI if
                          PySide6 is unavailable)
  --bootstrap             Install all required dependencies into .venv and .venv-t
  --bootstrap-ft          Install only free-threaded dependencies into .venv-t
  --help, -h              Show this help

Environment:
  QT_QPA_PLATFORM=...     Override Qt platform plugin (wayland / xcb / offscreen)
  OLLAMA_URL=...          Override Ollama server URL

Examples:
  $0                              Launch GUI
  $0 --cli --list-modes            List analysis modes
  $0 --cli --mode lowlevel --dry   Dry run on the system venv
  $0 --cli --ft --mode lowlevel --dry
                                  Dry run on the free-threaded venv (true parallel
                                  threads, no GIL)
  OLLAMA_URL=http://gpu:11434 $0 --cli --mode ollama
                                  Use a remote Ollama server

Notes:
  - CPU-heavy work uses ProcessPoolExecutor so it always scales with cores.
  - I/O (Ollama, file reading) uses threads so the GUI stays responsive.
  - Free-threaded Python is downloaded on first run (~30 MB compressed).
  - After 'pip install .', use the 'wanalyzer' and 'wanalyzer-gui' commands.
EOF
        exit 0
        ;;

    *)
        # Default: GUI
        py="$(venv_python "$VENV_DIR")"
        if [ -z "$py" ]; then
            bootstrap
            py="$VENV_DIR/bin/python"
        fi
        if ! have_mod "$py" PySide6; then
            warn "PySide6 not installed in $VENV_DIR; running bootstrap ..."
            bootstrap
        fi
        if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ] && [ "$QT_QPA_PLATFORM" = "offscreen" ]; then
            warn "No display detected and QT_QPA_PLATFORM=offscreen. GUI may not work."
            warn "Try CLI mode instead: $0 --cli --help"
        fi
        exec "$py" -m wallpaper_analyzer.gui "$@"
        ;;
esac
