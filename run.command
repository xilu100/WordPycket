#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
RUNTIME_DIR="$ROOT/runtime"
UV_DIR="$RUNTIME_DIR/uv"
UV_EXE="$UV_DIR/uv"
UV_ARCHIVE="$RUNTIME_DIR/uv.tar.gz"
PYTHON_INSTALL_DIR="$RUNTIME_DIR/python"
PYTHON_VERSION="3.11"
VENV_PATH="$ROOT/.venv"
VENV_PYTHON="$VENV_PATH/bin/python"
SETUP_MARKER="$RUNTIME_DIR/.setup-complete"

export PIP_CACHE_DIR="$RUNTIME_DIR/cache/pip"
export HF_HOME="$RUNTIME_DIR/cache/huggingface"
export XDG_CACHE_HOME="$RUNTIME_DIR/cache"
export NLTK_DATA="$RUNTIME_DIR/nltk_data"
export UV_CACHE_DIR="$RUNTIME_DIR/cache/uv"
export UV_PYTHON_INSTALL_DIR="$PYTHON_INSTALL_DIR"
export UV_PYTHON_PREFERENCE="only-managed"
export UV_PROJECT_ENVIRONMENT="$VENV_PATH"

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

install_local_uv
install_local_python

needs_setup=0
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

if [ ! -f "$SETUP_MARKER" ]; then
    needs_setup=1
fi

if [ "$needs_setup" -eq 1 ]; then
    "$VENV_PYTHON" "$ROOT/scripts/setup_env.py"
    touch "$SETUP_MARKER"
fi

"$VENV_PYTHON" -m wordpycket.main
