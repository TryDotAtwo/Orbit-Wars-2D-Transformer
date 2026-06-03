param(
    [Parameter(Mandatory = $true)]
    [string]$RunId,

    [Parameter(Mandatory = $true)]
    [int[]]$Generations,

    [string]$Workspace = (Get-Location).Path,

    [int]$PollSeconds = 15,

    [int]$StableChecks = 2
)

$ErrorActionPreference = "Stop"

foreach ($generation in $Generations) {
    $path = Join-Path $Workspace "dashboard/public/telemetry/replays_${RunId}_generation_${generation}.json"
    $previousLength = -1
    $stableCount = 0
    while ($true) {
        if (Test-Path -LiteralPath $path) {
            $length = (Get-Item -LiteralPath $path).Length
            if ($length -eq $previousLength -and $length -gt 0) {
                $stableCount += 1
            } else {
                $stableCount = 0
                $previousLength = $length
            }
            if ($stableCount -ge $StableChecks) {
                & node (Join-Path $Workspace "tools/convert_replay_chunk_v2.mjs") $path
                break
            }
        }
        Start-Sleep -Seconds $PollSeconds
    }
}
