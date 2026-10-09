# HANDOFF_GPU.md section 6: one detached analysis chain per PTM cache.
# Usage (PowerShell): .\launch_ptm_chains.ps1                    # all six, steps 1-4
#                     .\launch_ptm_chains.ps1 -Names imr,cub      # subset
#                     .\launch_ptm_chains.ps1 -Names ima -Steps 3,4   # only steps 3-4
# Steps: 1 baselines, 2 --only-cncm, 3 --adolescence --adol-mode protos, 4 lifecycle.
param([string[]]$Names = @("imr","ima","cub","obj","omni","vtab"),
      [string[]]$Steps = @("1","2","3","4"))
$ErrorActionPreference = "Stop"
# when invoked via powershell.exe -File, "imr,cub" arrives as ONE string: split it
$Names = @($Names | ForEach-Object { "$_" -split "," } | Where-Object { $_ })
$Steps = @($Steps | ForEach-Object { "$_" -split "," } | Where-Object { $_ } | ForEach-Object { [int]$_ })
$HERE = Split-Path -Parent $MyInvocation.MyCommand.Path
$W  = (Resolve-Path "$HERE\..").Path
$PY = "$HERE\venv\Scripts\python.exe"
if (-not (Test-Path $PY)) { throw "venv missing: $PY" }
New-Item -ItemType Directory -Force "$W\logs" | Out-Null
foreach ($name in $Names) {
  $tag = "${name}_vitb"
  if (-not (Test-Path "$W\results\ptm_$tag.npz")) { "skip $tag (no cache)"; continue }
  $cmds = @{
    1 = "& '$PY' -u exp_cifar.py --feats results\ptm_$tag.npz --tag $tag --prefix ptm --first-seed 2 --seeds 3";
    2 = "& '$PY' -u exp_cifar.py --feats results\ptm_$tag.npz --tag $tag --prefix ptm --first-seed 2 --seeds 3 --only-cncm";
    3 = "& '$PY' -u exp_cifar.py --feats results\ptm_$tag.npz --tag $tag --prefix ptm --adolescence --adol-mode protos --first-seed 2 --seeds 3";
    4 = "& '$PY' -u exp_lifecycle.py --benchmarks ptm_$tag --arbiter gated --gate repaired --first-seed 2 --seeds 3 --out-name lifecycle_ptm_$tag.json"
  }
  $cmd = "Set-Location '$W'; " + (($Steps | ForEach-Object { $cmds[[int]$_] }) -join "; ")
  Start-Process powershell.exe -ArgumentList @("-NoProfile","-Command",$cmd) -WorkingDirectory $W `
    -RedirectStandardOutput "$W\logs\ptm_$tag.log" -RedirectStandardError "$W\logs\ptm_$tag.err" -WindowStyle Hidden
  "launched $tag steps $($Steps -join ',')"
}
