# GPU-box payload #3: first-session tuning (Phase 1) caches, per dataset and seed.
# Trains parallel bottleneck adapters on that seed's FIRST task only, freezes
# them, and extracts the whole dataset -> out\<prefix>_<tag>_p1s<seed>.npz.
# Idempotent: an existing output is skipped, and a failed (dataset, seed) is
# reported and skipped rather than aborting the battery.
#
# Each call runs via Start-Process -Wait with stdout -> out\phase1_<name>_s<s>.log
# and stderr -> out\phase1_<name>_s<s>.err. Do NOT pipe python through
# `2>&1 | Tee-Object` here: under $ErrorActionPreference = "Stop" on Windows
# PowerShell 5.1 the first line torch writes to stderr becomes a terminating
# NativeCommandError and the whole battery aborts (this happened on 2026-09-08;
# the GPU box wrote run_phase1_box.ps1 to work around it, and that fix is now
# this script).
#
# Usage (from this folder):  .\run_phase1.ps1                # seeds 2,3,4; all present
#                            .\run_phase1.ps1 -Seeds 2 -Names cifar100,imr
param(
    [string[]]$Seeds = @("2","3","4"),
    [string[]]$Names = @(),
    [string]$Backbone = "vit_base_patch16_224.augreg_in21k",
    [string]$Tag = "vitb",
    [int]$Epochs = 20
)
$ErrorActionPreference = "Stop"
$Seeds = @($Seeds | ForEach-Object { "$_" -split "," } | Where-Object { $_ } | ForEach-Object { [int]$_ })
$Names = @($Names | ForEach-Object { "$_" -split "," } | Where-Object { $_ })
$HERE = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $HERE
$PY = "$HERE\venv\Scripts\python.exe"
if (-not (Test-Path $PY)) { throw "run .\setup_env.ps1 first" }
New-Item -ItemType Directory -Force "$HERE\out" | Out-Null

$SETS = @(
    @{ name = "cifar100"; dir = $null;            inc = 10; prefix = "cifar100" },
    @{ name = "imr";      dir = "imagenet-r";     inc = 20; prefix = "ptm" },
    @{ name = "ima";      dir = "imagenet-a";     inc = 20; prefix = "ptm" },
    @{ name = "cub";      dir = "cub";            inc = 20; prefix = "ptm" },
    @{ name = "obj";      dir = "objectnet";      inc = 20; prefix = "ptm" },
    @{ name = "omni";     dir = "omnibenchmark";  inc = 30; prefix = "ptm" },
    @{ name = "vtab";     dir = "vtab";           inc = 10; prefix = "ptm" }
)
$failed = @()
foreach ($d in $SETS) {
    if ($Names.Count -gt 0 -and $Names -notcontains $d.name) { continue }
    if ($d.dir -and -not (Test-Path "$HERE\data\$($d.dir)\train")) { Write-Host "skip $($d.name): data missing"; continue }
    foreach ($s in $Seeds) {
        $out = if ($d.prefix -eq "cifar100") { "$HERE\out\cifar100_${Tag}_p1s$s.npz" } else { "$HERE\out\ptm_$($d.name)_${Tag}_p1s$s.npz" }
        Write-Host "`n=== Phase 1: $($d.name) seed $s -> $out ==="
        if (Test-Path $out) { Write-Host "$out exists -- skipping"; continue }
        $args = @("`"$HERE\phase1_extract.py`"", "--dataset", $d.name, "--seed", $s, "--backbone", $Backbone, "--epochs", $Epochs, "--out", "`"$out`"")
        if ($d.dir) { $args += @("--root", "`"$HERE\data\$($d.dir)`"", "--inc", $d.inc) }
        # 2026-09-15: the log name carries the backbone tag. Without it the
        # battery-4 Block E runs (-Tag vitb1k) overwrote the battery-3 in21k
        # Phase-1 training logs for seeds 2-4 on the box (BATTERY4_RESULTS.md
        # section 7; caches and results were never at risk, they carry the tag).
        $log = "$HERE\out\phase1_$($d.name)_${Tag}_s$s.log"; $err = "$HERE\out\phase1_$($d.name)_${Tag}_s$s.err"
        $t0 = Get-Date
        $p = Start-Process -FilePath $PY -ArgumentList $args -WorkingDirectory $HERE -NoNewWindow -Wait -PassThru `
             -RedirectStandardOutput $log -RedirectStandardError $err
        $dt = [int]((Get-Date) - $t0).TotalSeconds
        if ($p.ExitCode -ne 0 -or -not (Test-Path $out)) {
            Write-Host "FAILED $($d.name) seed $s (exit $($p.ExitCode), ${dt}s) -- see $err"
            $failed += "$($d.name)_s$s"
        } else {
            Write-Host ("done {0} seed {1} in {2}s: " -f $d.name, $s, $dt) -NoNewline
            Get-Content $log | Select-String "task-0 val accuracy|^wrote" | ForEach-Object { Write-Host ("  " + $_.Line.Trim()) }
        }
    }
}
Write-Host "`nDone. Failed: $(if ($failed) { $failed -join ', ' } else { 'none' }). Copy $HERE\out\*_p1s*.npz to ..\results\ and follow HANDOFF_GPU2.md section 4."
