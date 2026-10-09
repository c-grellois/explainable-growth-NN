# GPU-box payload #2: the standard pretrained-model class-IL suite.
# Extracts ViT-B/16-in21k features (the backbone SimpleCIL / APER / RanPAC
# report on) for the six datasets those papers use, in their B0 protocols.
#
# Before running, place each dataset under .\data\<name>\{train,test}\<class>\*
# (the layout distributed with the RevisitingCIL / LAMDA-PILOT code; see
# README.md "Suite datasets"). Any dataset whose folder is missing is skipped,
# and any extraction whose output exists is skipped, so this is re-runnable.
#
# Usage (PowerShell, from this folder, after setup_and_extract.ps1 built .\venv):
#   .\extract_suite.ps1                # ViT-B/16-in21k, all six
#   .\extract_suite.ps1 -Backbone vit_base_patch14_dinov2.lvd142m -Tag dinov2b

param(
    [string]$Backbone = "vit_base_patch16_224.augreg_in21k",
    [string]$Tag = "vitb",
    [string[]]$Names = @()      # subset of imr ima cub obj omni vtab; empty = all present
)
$ErrorActionPreference = "Stop"
$HERE = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $HERE
$PY = "$HERE\venv\Scripts\python.exe"
if (-not (Test-Path $PY)) { throw "run .\setup_and_extract.ps1 first (creates .\venv)" }
New-Item -ItemType Directory -Force "$HERE\out" | Out-Null

# name, folder, classes-per-task (B0 protocol of Zhou et al. 2023 / RanPAC)
$SUITE = @(
    @{ name = "imr";  dir = "imagenet-r";    inc = 20 },   # ImageNet-R      200 classes, 10 tasks
    @{ name = "ima";  dir = "imagenet-a";    inc = 20 },   # ImageNet-A      200 classes, 10 tasks
    @{ name = "cub";  dir = "cub";           inc = 20 },   # CUB-200         200 classes, 10 tasks
    @{ name = "obj";  dir = "objectnet";     inc = 20 },   # ObjectNet-200   200 classes, 10 tasks
    @{ name = "omni"; dir = "omnibenchmark"; inc = 30 },   # OmniBenchmark   300 classes, 10 tasks
    @{ name = "vtab"; dir = "vtab";          inc = 10 }    # VTAB            50 classes,  5 tasks
)

foreach ($d in $SUITE) {
    if ($Names.Count -gt 0 -and $Names -notcontains $d.name) { continue }
    $root = "$HERE\data\$($d.dir)"
    $out  = "$HERE\out\ptm_$($d.name)_$Tag.npz"
    if (-not (Test-Path "$root\train")) { Write-Host "skip $($d.name): $root\train not found"; continue }
    Write-Host "`n=== $($d.name)  ($root)  inc $($d.inc)  -> $out ==="
    & $PY "$HERE\extract_ptm.py" --root $root --dataset $d.name --inc $d.inc --backbone $Backbone --out $out
}

Write-Host "`nDone. Copy $HERE\out\ptm_*.npz to ASDplusContracts\results\ on the analysis machine,"
Write-Host "then per cache:  exp_cifar.py --feats results\ptm_<name>_$Tag.npz --tag <name>_$Tag --prefix ptm --first-seed 2 --seeds 3"
