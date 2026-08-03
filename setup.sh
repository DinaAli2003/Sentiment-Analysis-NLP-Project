#!/usr/bin/env bash
#
# setup.sh — one-time environment setup for macOS / Linux / VS Code.
#
# Creates a Python 3.11 virtual environment (.venv) and installs every pinned
# dependency from requirements.txt.
#
# Python 3.11 is used deliberately: gensim and hmmlearn ship compiled
# C/Cython extensions that don't yet reliably have prebuilt wheels for the
# newest Python releases everywhere. Python 3.11 is the tested, safe zone
# for every package pinned in requirements.txt.
#
# Usage (from the project root):
#     bash setup.sh
#
# Then in VS Code: Ctrl+Shift+P -> "Python: Select Interpreter" (or
# "Notebook: Select Kernel") -> choose .venv/bin/python

set -euo pipefail

echo "== Sentiment Analysis project setup =="

PYTHON_BIN=""
for candidate in python3.11 python3.12 python3.10; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PYTHON_BIN="$candidate"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo ""
    echo "Python 3.11 (or 3.10/3.12) was not found on PATH."
    echo "Install it (e.g. 'brew install python@3.11' on macOS, or via your"
    echo "distro's package manager on Linux), then re-run this script."
    exit 1
fi
echo "Using $($PYTHON_BIN --version)"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment (.venv)..."
    "$PYTHON_BIN" -m venv .venv
else
    echo ".venv already exists, reusing it."
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "Upgrading pip/setuptools/wheel..."
python -m pip install --upgrade pip setuptools wheel

echo "Installing pinned dependencies from requirements.txt (this can take several minutes)..."
pip install -r requirements.txt

echo ""
echo "Setup complete."
echo "Next steps:"
echo "  1. In VS Code: Ctrl+Shift+P -> 'Python: Select Interpreter' -> choose .venv"
echo "  2. Open notebooks/Full_Pipeline.ipynb and select the same .venv as the kernel"
echo "  3. Run all cells top to bottom"
