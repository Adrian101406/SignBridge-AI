$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$uvCommand = Get-Command uv -ErrorAction Stop

Push-Location $projectRoot
try {
    & $uvCommand.Source python install 3.11
    if ($LASTEXITCODE -ne 0) { throw "uv could not install Python 3.11" }

    & $uvCommand.Source sync --frozen
    if ($LASTEXITCODE -ne 0) { throw "uv could not synchronize the environment" }

    & $uvCommand.Source run python main.py --check --skip-models
    if ($LASTEXITCODE -ne 0) { throw "SignBridge preflight failed" }
}
finally {
    Pop-Location
}
