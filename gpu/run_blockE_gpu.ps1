# HANDOFF_GPU3.md section 6, GPU part of Block E (optional): the ImageNet-1K
# checkpoint for the three suite datasets not yet extracted on it, then
# first-session adapters (Phase 1) on the 1k checkpoint for all six datasets,
# seeds 2-4. Sequential; every step is idempotent (existing outputs skipped).
# Run as a child process:  powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_blockE_gpu.ps1
$HERE = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $HERE
$PY = "$HERE\venv\Scripts\python.exe"
$BB = "vit_base_patch16_224.augreg_in21k_ft_in1k"
Write-Host "=== Block E / 1: in1k caches for obj, omni, vtab ==="
& .\extract_suite.ps1 -Backbone $BB -Tag vitb1k -Names obj,omni,vtab
Write-Host "=== Block E / 2: Phase 1 on the in1k checkpoint, six datasets, seeds 2-4 ==="
& .\run_phase1.ps1 -Backbone $BB -Tag vitb1k -Names imr,ima,cub,obj,omni,vtab -Seeds 2,3,4
Write-Host "=== Block E / 3: cache summary ==="
& $PY cache_summary.py (Get-ChildItem "$HERE\out\ptm_*_vitb1k*.npz" | ForEach-Object { $_.FullName })
Write-Host "BLOCK E GPU DONE"
