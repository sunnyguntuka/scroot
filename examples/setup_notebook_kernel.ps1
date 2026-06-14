# Creates .venv at the repo root and registers a Jupyter kernel for the demo notebooks.
# Usage:  powershell -ExecutionPolicy Bypass -File examples/setup_notebook_kernel.ps1
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $RepoRoot ".venv"
$Requirements = Join-Path $PSScriptRoot "requirements-notebook.txt"

# Find a Python 3.9+ interpreter: prefer the py launcher, fall back to python.
if (Get-Command py -ErrorAction SilentlyContinue) {
    $BasePython = "py"
    $BaseArgs = @("-3")
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $BasePython = "python"
    $BaseArgs = @()
} else {
    throw "Python not found. Install Python 3.9+ from https://python.org and try again."
}

if (-not (Test-Path $VenvPath)) {
    Write-Host "Creating virtual environment at $VenvPath ..."
    & $BasePython @BaseArgs -m venv $VenvPath
}

$Python = Join-Path $VenvPath "Scripts\python.exe"

Write-Host "Upgrading pip ..."
& $Python -m pip install --upgrade pip --quiet

Write-Host "Installing scroot (editable) with notebook extras ..."
& $Python -m pip install -e "${RepoRoot}[pandas,database,security]" --quiet

Write-Host "Installing notebook dependencies ..."
& $Python -m pip install -r $Requirements --quiet

Write-Host "Registering Jupyter kernel 'scroot-demo' ..."
& $Python -m ipykernel install --user --name scroot-demo --display-name "Python (scroot-demo)"

Write-Host ""
Write-Host "Done. Two ways to open the demo:"
Write-Host "  1. VS Code / Cursor: open examples\scroot_interactive_demo.ipynb and select kernel 'Python (scroot-demo)'"
Write-Host "  2. Browser:          .venv\Scripts\jupyter notebook examples\scroot_interactive_demo.ipynb"
