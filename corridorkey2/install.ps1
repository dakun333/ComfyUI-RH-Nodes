param(
    [string]$PythonExe = '',
    [switch]$Check,
    [switch]$DownloadModels,
    [switch]$Verify,
    [string]$IndexUrl = ''
)
$ErrorActionPreference = 'Stop'
if (-not $PythonExe) {
    $comfyRoot = Split-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) -Parent
    $candidates = @(
        (Join-Path $comfyRoot '.venv\Scripts\python.exe'),
        (Join-Path (Split-Path $comfyRoot -Parent) 'python_embeded\python.exe')
    )
    $PythonExe = $candidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
}
if (-not $PythonExe -or -not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw 'Cannot locate ComfyUI Python. Pass -PythonExe with the exact Python used to start ComfyUI.'
}
$env:PYTHONIOENCODING = 'utf-8'
$arguments = @((Join-Path $PSScriptRoot 'install.py'))
if ($Check) { $arguments += '--check' }
if ($DownloadModels) { $arguments += '--download-models' }
if ($Verify) { $arguments += '--verify' }
if ($IndexUrl) { $arguments += @('--index-url', $IndexUrl) }
& $PythonExe @arguments
if ($LASTEXITCODE -ne 0) { throw "CorridorKey2 installation/check failed (exit $LASTEXITCODE)." }
