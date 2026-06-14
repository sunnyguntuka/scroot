#!/usr/bin/env bash
# Creates .venv at the repo root and registers a Jupyter kernel for the demo notebooks.
# Usage:  bash examples/setup_notebook_kernel.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PATH="$REPO_ROOT/.venv"
REQUIREMENTS="$REPO_ROOT/examples/requirements-notebook.txt"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found. Install Python 3.9+ and try again." >&2
  exit 1
fi

if [[ ! -d "$VENV_PATH" ]]; then
  echo "Creating virtual environment at $VENV_PATH ..."
  python3 -m venv "$VENV_PATH"
fi

PYTHON="$VENV_PATH/bin/python"

echo "Upgrading pip ..."
"$PYTHON" -m pip install --upgrade pip --quiet

echo "Installing scroot (editable) with notebook extras ..."
"$PYTHON" -m pip install -e "$REPO_ROOT[pandas,database,security]" --quiet

echo "Installing notebook dependencies ..."
"$PYTHON" -m pip install -r "$REQUIREMENTS" --quiet

echo "Registering Jupyter kernel 'scroot-demo' ..."
"$PYTHON" -m ipykernel install --user --name scroot-demo --display-name "Python (scroot-demo)"

echo ""
echo "Done. Two ways to open the demo:"
echo "  1. VS Code / Cursor: open examples/scroot_interactive_demo.ipynb and select kernel 'Python (scroot-demo)'"
echo "  2. Browser:          .venv/bin/jupyter notebook examples/scroot_interactive_demo.ipynb"
