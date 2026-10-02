$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runtimeCache = Join-Path $projectRoot ".runtime-cache"
$uvCommand = Get-Command uv -ErrorAction Stop

New-Item -ItemType Directory -Force -Path $runtimeCache | Out-Null
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_DATASETS_OFFLINE = "1"
$env:HF_HOME = $runtimeCache
$env:TRANSFORMERS_CACHE = Join-Path $runtimeCache "transformers"
$env:USE_TF = "0"

Push-Location $projectRoot
try {
    & $uvCommand.Source run python main.py --check
    if ($LASTEXITCODE -ne 0) { throw "SignBridge preflight failed" }
    & $uvCommand.Source run python main.py
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
