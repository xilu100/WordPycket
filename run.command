#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
RUNTIME_DIR="$ROOT/runtime"
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/wordpycket-$(date +%Y%m%d-%H%M%S).log"
UV_DIR="$RUNTIME_DIR/uv"
UV_EXE="$UV_DIR/uv"
UV_ARCHIVE="$RUNTIME_DIR/uv.tar.gz"
PYTHON_INSTALL_DIR="$RUNTIME_DIR/python"
PYTHON_VERSION="3.11"
VENV_PATH="$ROOT/.venv"
VENV_PYTHON="$VENV_PATH/bin/python"
SETUP_MARKER="$RUNTIME_DIR/.setup-complete"
SETUP_SCRIPT="$ROOT/scripts/setup_env.py"
CHECK_SCRIPT="$ROOT/scripts/check_env.py"
PYPROJECT_FILE="$ROOT/pyproject.toml"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1
trap 'status=$?; if [ "$status" -ne 0 ]; then echo; echo "WordPycket failed to start."; echo "Startup log saved to: $LOG_FILE"; fi' EXIT
echo "WordPycket startup log: $LOG_FILE"

export PIP_CACHE_DIR="$RUNTIME_DIR/cache/pip"
export HF_HOME="$RUNTIME_DIR/cache/huggingface"
export XDG_CACHE_HOME="$RUNTIME_DIR/cache"
export NLTK_DATA="$RUNTIME_DIR/nltk_data"
export UV_CACHE_DIR="$RUNTIME_DIR/cache/uv"
export UV_PYTHON_INSTALL_DIR="$PYTHON_INSTALL_DIR"
export UV_PYTHON_PREFERENCE="only-managed"
export UV_PROJECT_ENVIRONMENT="$VENV_PATH"
if [ -n "${PYTHONPATH:-}" ]; then
    export PYTHONPATH="$ROOT/src:$PYTHONPATH"
else
    export PYTHONPATH="$ROOT/src"
fi

mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$NLTK_DATA" "$UV_CACHE_DIR" "$UV_DIR"
mkdir -p "$ROOT/input" "$ROOT/model" "$ROOT/data/csv_databases"

install_local_uv() {
    if [ -x "$UV_EXE" ]; then
        return
    fi

    case "$(uname -m)" in
        arm64|aarch64)
            uv_url="https://github.com/astral-sh/uv/releases/latest/download/uv-aarch64-apple-darwin.tar.gz"
            ;;
        x86_64|amd64)
            uv_url="https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-apple-darwin.tar.gz"
            ;;
        *)
            echo "Unsupported macOS CPU architecture: $(uname -m)"
            exit 1
            ;;
    esac

    echo "Installing local uv bootstrapper into $UV_DIR"
    curl -L "$uv_url" -o "$UV_ARCHIVE"
    tar -xzf "$UV_ARCHIVE" -C "$UV_DIR" --strip-components=1
    rm -f "$UV_ARCHIVE"
    chmod +x "$UV_EXE"
}

install_local_python() {
    echo "Installing managed Python $PYTHON_VERSION into $PYTHON_INSTALL_DIR"
    "$UV_EXE" python install "$PYTHON_VERSION"
}

get_venv_python_version() {
    if [ ! -x "$VENV_PYTHON" ]; then
        return 1
    fi
    "$VENV_PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
}

file_hash() {
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        openssl dgst -sha256 "$1" | awk '{print $NF}'
    fi
}

setup_fingerprint() {
    printf 'python=%s\nsetup=2\n' "$PYTHON_VERSION"
    for path in "$PYPROJECT_FILE" "$SETUP_SCRIPT" "$CHECK_SCRIPT"; do
        name="$(basename "$path")"
        if [ -f "$path" ]; then
            printf '%s=%s\n' "$name" "$(file_hash "$path")"
        else
            printf '%s=missing\n' "$name"
        fi
    done
}

setup_marker_matches() {
    expected="$1"
    if [ ! -f "$SETUP_MARKER" ]; then
        return 1
    fi
    [ "$(cat "$SETUP_MARKER")" = "$expected" ]
}

install_local_uv
install_local_python

needs_setup=0
setup_hash="$(setup_fingerprint)"
existing_python_version="$(get_venv_python_version || true)"
if [ -n "$existing_python_version" ] && [ "$existing_python_version" != "$PYTHON_VERSION" ]; then
    echo "Existing virtual environment uses Python $existing_python_version; recreating with Python $PYTHON_VERSION."
    rm -rf "$VENV_PATH"
    rm -f "$SETUP_MARKER"
    needs_setup=1
fi

if [ ! -x "$VENV_PYTHON" ]; then
    "$UV_EXE" venv --seed --python "$PYTHON_VERSION" "$VENV_PATH"
    needs_setup=1
fi

if ! setup_marker_matches "$setup_hash"; then
    needs_setup=1
fi

if [ "$needs_setup" -eq 1 ]; then
    "$VENV_PYTHON" "$SETUP_SCRIPT" --strict-accel
    printf '%s' "$setup_hash" > "$SETUP_MARKER"
else
    if ! "$VENV_PYTHON" "$CHECK_SCRIPT"; then
        echo "Environment check found missing or outdated components; repairing environment."
        "$VENV_PYTHON" "$SETUP_SCRIPT" --strict-accel
        printf '%s' "$setup_hash" > "$SETUP_MARKER"
    fi
fi

"$VENV_PYTHON" -m wordpycket.main
