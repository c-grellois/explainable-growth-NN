# Build the GPU-box venv for BOTH jobs: feature extraction (CUDA torch + timm)
# and the CPU analysis pipeline (numpy, scipy, scikit-learn, matplotlib).
# Idempotent. Run from this folder in PowerShell:
#   Set-ExecutionPolicy -Scope Process Bypass; .\setup_env.ps1

$ErrorActionPreference = "Stop"
$HERE = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $HERE
New-Item -ItemType Directory -Force "$HERE\out"  | Out-Null
New-Item -ItemType Directory -Force "$HERE\data" | Out-Null

if (-not (Test-Path "$HERE\venv")) {
    python -m venv "$HERE\venv"
}
$PY = "$HERE\venv\Scripts\python.exe"
& $PY -m pip install --upgrade pip
& $PY -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
& $PY -m pip install timm numpy scipy scikit-learn matplotlib tqdm
& $PY -c "import torch, timm, sklearn, scipy, numpy; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '(none)'); print('timm', timm.__version__, '| sklearn', sklearn.__version__, '| numpy', numpy.__version__)"
# the interpreter guard in cgrow/__init__.py must accept this venv
& $PY -c "import sys, os; sys.path.insert(0, os.path.join(r'$HERE', '..')); import cgrow; print('cgrow guard OK under', sys.executable)"
Write-Host "`nvenv ready: $PY"
