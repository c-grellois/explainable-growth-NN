# One detached CPU analysis chain per (cache, seed-block), any cache name.
# Generalises launch_ptm_chains.ps1 (HANDOFF_GPU.md section 6) to arbitrary
# caches and seeds, and to the seed-specific Phase-1 caches (one seed each).
#
#   .\launch_chains.ps1 -Caches ptm_banking77_mpnet,ptm_clinc150_mpnet          # seeds 2,3,4 in one chain
#   .\launch_chains.ps1 -Caches ptm_nslkdd_raw -Steps 1,2,5
#   .\launch_chains.ps1 -Caches cifar100_vitb_p1s5,ptm_imr_vitb_p1s5 -PerCacheSeed   # seed taken from _p1s<k>
#   .\launch_chains.ps1 -Caches ptm_imr_vitb1k -Seeds 5,6,7 -OnePerSeed          # three chains, one seed each
#
# Steps: 1 baselines (growth, finetune, EWC, replay, NCM, joint)
#        2 --only-cncm (contracted prototypes; needs step 1's json)
#        3 --adolescence --adol-mode protos (caches the organism)
#        4 certified lifecycle (reuses the organism)
#        5 contracted head, M = 2000 and 10000
# The prefix (cifar100 vs ptm) and the tag are derived from the cache name.
param([string[]]$Caches = @(),
      [string[]]$Seeds = @("2","3","4"),
      [string[]]$Steps = @("1","2","3","4","5"),
      [switch]$PerCacheSeed,
      [switch]$OnePerSeed)
$ErrorActionPreference = "Stop"
$Caches = @($Caches | ForEach-Object { "$_" -split "," } | Where-Object { $_ })
$Seeds  = @($Seeds  | ForEach-Object { "$_" -split "," } | Where-Object { $_ } | ForEach-Object { [int]$_ })
$Steps  = @($Steps  | ForEach-Object { "$_" -split "," } | Where-Object { $_ } | ForEach-Object { [int]$_ })
if ($Caches.Count -eq 0) { throw "give -Caches" }
$HERE = Split-Path -Parent $MyInvocation.MyCommand.Path
$W  = (Resolve-Path "$HERE\..").Path
$PY = "$HERE\venv\Scripts\python.exe"
if (-not (Test-Path $PY)) { throw "venv missing: $PY" }
New-Item -ItemType Directory -Force "$W\logs" | Out-Null

function Chain($cache, $first, $n) {
  if ($cache.StartsWith("cifar100_")) { $prefix = "cifar100"; $tag = $cache.Substring(9); $bench = "cifar_$tag" }
  elseif ($cache.StartsWith("ptm_"))  { $prefix = "ptm";      $tag = $cache.Substring(4); $bench = "ptm_$tag" }
  else { throw "cache name must start with cifar100_ or ptm_: $cache" }
  $feats = "results\$cache.npz"
  $c = @{
    1 = "& '$PY' -u exp_cifar.py --feats $feats --tag $tag --prefix $prefix --first-seed $first --seeds $n";
    2 = "& '$PY' -u exp_cifar.py --feats $feats --tag $tag --prefix $prefix --first-seed $first --seeds $n --only-cncm";
    3 = "& '$PY' -u exp_cifar.py --feats $feats --tag $tag --prefix $prefix --adolescence --adol-mode protos --first-seed $first --seeds $n";
    4 = "& '$PY' -u exp_lifecycle.py --benchmarks $bench --arbiter gated --gate repaired --first-seed $first --seeds $n --out-name lifecycle_$bench" + $(if ($n -eq 1) { "_s$first" } else { "" }) + ".json";
    5 = "& '$PY' -u exp_head.py --cache $cache --M 2000 10000 --first-seed $first --seeds $n"
  }
  return "Set-Location '$W'; " + (($Steps | ForEach-Object { $c[[int]$_] }) -join "; ")
}

foreach ($cache in $Caches) {
  if (-not (Test-Path "$W\results\$cache.npz")) { "skip $cache (no cache)"; continue }
  $blocks = @()
  if ($PerCacheSeed) {
    if ($cache -notmatch "_p1s(\d+)$") { throw "-PerCacheSeed needs a _p1s<seed> cache: $cache" }
    $blocks += ,@([int]$Matches[1], 1)
  } elseif ($OnePerSeed) { foreach ($s in $Seeds) { $blocks += ,@($s, 1) } }
  else { $blocks += ,@($Seeds[0], $Seeds.Count) }
  foreach ($b in $blocks) {
    $first, $n = $b
    $cmd = Chain $cache $first $n
    $stem = if ($n -eq 1) { "${cache}_s$first" } else { $cache }
    Start-Process powershell.exe -ArgumentList @("-NoProfile","-Command",$cmd) -WorkingDirectory $W `
      -RedirectStandardOutput "$W\logs\chain_$stem.log" -RedirectStandardError "$W\logs\chain_$stem.err" -WindowStyle Hidden
    "launched $stem  (seeds $first..$($first+$n-1); steps $($Steps -join ','))"
  }
}
"Check progress:  Get-Content logs\chain_<cache>.log -Tail 5 ; a non-empty .err means that chain died."
