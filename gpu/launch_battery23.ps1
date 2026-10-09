# HANDOFF_GPU2.md section 5: analysis chains for battery 2 (in1k caches) and
# battery 3 (Phase-1 caches). One detached chain per cache, logs in ..\logs\.
#
#   .\launch_battery23.ps1 -Kind vitb1k -Names imr,ima,cub                 # battery 2, seeds 2-4
#   .\launch_battery23.ps1 -Kind p1 -Names cifar100,imr -Seeds 2,3,4       # battery 3, one chain per (dataset, seed)
#   ... -Steps 2,3          # subset of steps: 1 baselines, 2 --only-cncm, 3 exp_head, 4 adolescence, 5 lifecycle
param(
  [ValidateSet("vitb1k","p1")][string]$Kind = "vitb1k",
  [string[]]$Names = @("imr","ima","cub"),
  [string[]]$Seeds = @("2","3","4"),
  [string[]]$Steps = @("1","2","3","4","5"),
  [int]$Threads = 2
)
$ErrorActionPreference = "Stop"
$Names = @($Names | ForEach-Object { "$_" -split "," } | Where-Object { $_ })
$Seeds = @($Seeds | ForEach-Object { "$_" -split "," } | Where-Object { $_ } | ForEach-Object { [int]$_ })
$Steps = @($Steps | ForEach-Object { "$_" -split "," } | Where-Object { $_ } | ForEach-Object { [int]$_ })
$HERE = Split-Path -Parent $MyInvocation.MyCommand.Path
$W  = (Resolve-Path "$HERE\..").Path
$PY = "$HERE\venv\Scripts\python.exe"
if (-not (Test-Path $PY)) { throw "venv missing: $PY" }
New-Item -ItemType Directory -Force "$W\logs" | Out-Null

function Launch($cache, $tag, $prefixArg, $bench, $first, $n, $logname) {
  if (-not (Test-Path "$W\results\$cache.npz")) { "skip $cache (no cache)"; return }
  $cmds = @{
    1 = "& '$PY' -u exp_cifar.py --feats results\$cache.npz --tag $tag $prefixArg --first-seed $first --seeds $n";
    2 = "& '$PY' -u exp_cifar.py --feats results\$cache.npz --tag $tag $prefixArg --first-seed $first --seeds $n --only-cncm";
    3 = "& '$PY' -u exp_head.py --cache $cache --M 2000 10000 --first-seed $first --seeds $n";
    4 = "& '$PY' -u exp_cifar.py --feats results\$cache.npz --tag $tag $prefixArg --adolescence --adol-mode protos --first-seed $first --seeds $n";
    5 = "& '$PY' -u exp_lifecycle.py --benchmarks $bench --arbiter gated --gate repaired --first-seed $first --seeds $n --out-name lifecycle_$bench.json"
  }
  # cap BLAS/torch threads per chain (performance only; numerics unchanged) so
  # several CPU chains do not starve the GPU-side Phase 1 data loaders
  $env_ = "`$env:OMP_NUM_THREADS='$Threads'; `$env:MKL_NUM_THREADS='$Threads'; `$env:OPENBLAS_NUM_THREADS='$Threads'; "
  $cmd = "Set-Location '$W'; " + $env_ + (($Steps | ForEach-Object { $cmds[[int]$_] }) -join "; ")
  Start-Process powershell.exe -ArgumentList @("-NoProfile","-Command",$cmd) -WorkingDirectory $W `
    -RedirectStandardOutput "$W\logs\$logname.log" -RedirectStandardError "$W\logs\$logname.err" -WindowStyle Hidden
  "launched $logname steps $($Steps -join ',')"
}

if ($Kind -eq "vitb1k") {
  foreach ($name in $Names) {
    Launch "ptm_${name}_vitb1k" "${name}_vitb1k" "--prefix ptm" "ptm_${name}_vitb1k" 2 3 "ptm_${name}_vitb1k"
  }
} else {
  foreach ($name in $Names) {
    foreach ($s in $Seeds) {
      if ($name -eq "cifar100") {
        Launch "cifar100_vitb_p1s$s" "vitb_p1s$s" "" "cifar_vitb_p1s$s" $s 1 "cifar100_vitb_p1s$s"
      } else {
        Launch "ptm_${name}_vitb_p1s$s" "${name}_vitb_p1s$s" "--prefix ptm" "ptm_${name}_vitb_p1s$s" $s 1 "ptm_${name}_vitb_p1s$s"
      }
    }
  }
}
