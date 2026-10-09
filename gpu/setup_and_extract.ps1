# GPU-box payload: run this once on the RTX 3070 machine.
# Creates a CUDA venv, then extracts frozen-backbone feature caches for the
# contracted-growth benchmarks. Each extraction is resumable (skips if its
# output exists). Copy the produced gpu\out\*.npz back into
# ASDplusContracts\results\ on the main machine.
#
# Usage (PowerShell, from this folder):
#   .\setup_and_extract.ps1

$ErrorActionPreference = "Stop"
$HERE = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $HERE
New-Item -ItemType Directory -Force "$HERE\out" | Out-Null

# --- 1. venv + CUDA torch + timm ------------------------------------------
if (-not (Test-Path "$HERE\venv")) {
    python -m venv "$HERE\venv"
}
$PY = "$HERE\venv\Scripts\python.exe"
& $PY -m pip install --upgrade pip
& $PY -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
& $PY -m pip install timm numpy
& $PY -c "import torch; print('cuda available:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# --- 2. extractions ---------------------------------------------------------
# order: most valuable first. Each ~3-10 min on a 3070.
& $PY "$HERE\extract_gpu.py" --backbone vit_base_patch16_224.augreg_in21k --dataset cifar100 --per-class 500 --out "$HERE\out\cifar100_vitb_full.npz"
& $PY "$HERE\extract_gpu.py" --backbone vit_base_patch14_dinov2.lvd142m   --dataset cifar100 --per-class 500 --out "$HERE\out\cifar100_dinov2b.npz"
& $PY "$HERE\extract_gpu.py" --backbone vit_large_patch16_224.augreg_in21k --dataset cifar100 --per-class 500 --out "$HERE\out\cifar100_vitl.npz"
& $PY "$HERE\extract_gpu.py" --backbone vit_base_patch16_224.augreg_in21k --dataset cifar10  --per-class 500 --out "$HERE\out\cifar10_vitb.npz"

Write-Host "`nAll extractions done. Copy $HERE\out\*.npz to ASDplusContracts\results\ on the main machine."
