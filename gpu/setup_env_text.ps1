# Adds the text-substrate dependencies (HF datasets + transformers) and pandas
# (tabular streams) to the existing GPU-box venv. Idempotent.
#   Set-ExecutionPolicy -Scope Process Bypass; .\setup_env_text.ps1
$ErrorActionPreference = "Stop"
$HERE = Split-Path -Parent $MyInvocation.MyCommand.Path
$PY = "$HERE\venv\Scripts\python.exe"
if (-not (Test-Path $PY)) { throw "run .\setup_env.ps1 first" }
& $PY -m pip install "datasets>=2.19" "transformers>=4.40" sentencepiece pandas
& $PY -c "import datasets, transformers, pandas; print('datasets', datasets.__version__, '| transformers', transformers.__version__, '| pandas', pandas.__version__)"
Write-Host "text/tabular deps ready"
