<#
setup.ps1 — one-time environment setup for Windows / VS Code.

Creates a Python 3.11 virtual environment (.venv) and installs every pinned
dependency from requirements.txt.

Python 3.11 is used deliberately: gensim and hmmlearn ship compiled C/Cython
extensions that don't yet have prebuilt wheels for the newest Python releases
(3.13+) on Windows. Installing them there falls back to compiling from source,
which fails without a full Visual C++ build toolchain. Python 3.11 sidesteps
that entirely — every package in requirements.txt has a tested prebuilt wheel
for it.

Usage (from the project root, in PowerShell):
    .\setup.ps1

Then in VS Code: Ctrl+Shift+P -> "Python: Select Interpreter" (or
"Notebook: Select Kernel") -> choose .venv\Scripts\python.exe
#>

$ErrorActionPreference = "Stop"

Write-Host "== Sentiment Analysis project setup ==" -ForegroundColor Cyan

# 1. Locate a Python 3.11 interpreter via the Windows 'py' launcher
$pyCheck = & py -3.11 -c "print('ok')" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Python 3.11 was not found via the 'py' launcher." -ForegroundColor Yellow
    Write-Host "Install it from https://www.python.org/downloads/ (pick a 3.11.x release)," -ForegroundColor Yellow
    Write-Host "then re-run this script." -ForegroundColor Yellow
    exit 1
}
Write-Host "Found Python 3.11." -ForegroundColor Green

# 2. Create the virtual environment if it doesn't already exist
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment (.venv)..." -ForegroundColor Cyan
    py -3.11 -m venv .venv
} else {
    Write-Host ".venv already exists, reusing it." -ForegroundColor Yellow
}

# 3. Activate it for the rest of this script
& .\.venv\Scripts\Activate.ps1

# 4. Upgrade packaging tools, then install pinned requirements
Write-Host "Upgrading pip/setuptools/wheel..." -ForegroundColor Cyan
python -m pip install --upgrade pip setuptools wheel

Write-Host "Installing pinned dependencies from requirements.txt (this can take several minutes)..." -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. In VS Code: Ctrl+Shift+P -> 'Python: Select Interpreter' -> choose .venv"
Write-Host "  2. Open notebooks/Full_Pipeline.ipynb and select the same .venv as the kernel"
Write-Host "  3. Run all cells top to bottom"
