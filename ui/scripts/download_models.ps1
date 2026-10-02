$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$cacheRoot = Join-Path $projectRoot ".model-download-cache"
$whisperDir = Join-Path $projectRoot "models\whisper\malaysian-whisper-small-v3"
$mcieDir = Join-Path $projectRoot "models\mcie\Qwen3-4B-Instruct-2507"
$mcieGgufDir = Join-Path $projectRoot "models\mcie\Qwen3-4B-Instruct-2507-GGUF"
$uvCommand = Get-Command uv -ErrorAction Stop

$resolvedProject = [System.IO.Path]::GetFullPath($projectRoot).TrimEnd('\') + '\'
$resolvedCache = [System.IO.Path]::GetFullPath($cacheRoot)
if (-not $resolvedCache.StartsWith($resolvedProject, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to use a model cache outside the project: $resolvedCache"
}

New-Item -ItemType Directory -Force -Path $whisperDir, $mcieDir, $mcieGgufDir, $cacheRoot | Out-Null
$env:HF_HOME = $cacheRoot
$env:HUGGINGFACE_HUB_CACHE = Join-Path $cacheRoot "hub"

Push-Location $projectRoot
try {
    & $uvCommand.Source run huggingface-cli download mesolitica/malaysian-whisper-small-v3 `
        --revision ea0c0732303a436a8795dd00b4d844e4cb191b63 `
        --local-dir $whisperDir
    if ($LASTEXITCODE -ne 0) { throw "Malaysian Whisper download failed" }

    & $uvCommand.Source run huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 `
        --revision cdbee75f17c01a7cc42f958dc650907174af0554 `
        --local-dir $mcieDir
    if ($LASTEXITCODE -ne 0) { throw "Qwen download failed" }

    & $uvCommand.Source run huggingface-cli download unsloth/Qwen3-4B-Instruct-2507-GGUF `
        Qwen3-4B-Instruct-2507-Q4_K_M.gguf `
        --revision a06e946bb6b655725eafa393f4a9745d460374c9 `
        --local-dir $mcieGgufDir
    if ($LASTEXITCODE -ne 0) { throw "Qwen GGUF download failed" }

    & $uvCommand.Source run python -m signbridge_app.preflight
    if ($LASTEXITCODE -ne 0) { throw "Downloaded models did not pass preflight" }
}
finally {
    Pop-Location
}

if (Test-Path -LiteralPath $cacheRoot) {
    Remove-Item -LiteralPath $cacheRoot -Recurse -Force
}
