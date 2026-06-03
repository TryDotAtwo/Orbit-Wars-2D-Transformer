param(
    [string]$WorkspaceRoot = (Split-Path -Parent (Split-Path -Parent $PSCommandPath)),
    [string]$StageRoot = "",
    [string]$Competition = "orbit-wars",
    [string]$KaggleCommand = "kaggle",
    [int]$PollSeconds = 60,
    [int]$MinimumGeneration = 0,
    [switch]$ExitAfterFirstSubmit
)

$ErrorActionPreference = "Stop"

if ($StageRoot -eq "") {
    $StageRoot = Join-Path $WorkspaceRoot "artifacts\kaggle_auto_submit"
}

$logPath = Join-Path $WorkspaceRoot "artifacts\kaggle_host_submit_watcher.log"
$null = New-Item -ItemType Directory -Force -Path (Split-Path -Parent $logPath)

function Write-SubmitLog {
    param([string]$Message)
    $timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    Add-Content -Path $logPath -Value "$timestamp $Message"
}

function Get-GenerationNumber {
    param([string]$ReadyPath)
    $directoryName = Split-Path -Leaf (Split-Path -Parent $ReadyPath)
    if ($directoryName -match "^generation_(\d+)$") {
        return [int]$Matches[1]
    }
    return -1
}

function Submit-ReadyFile {
    param([System.IO.FileInfo]$ReadyFile)

    $generation = Get-GenerationNumber -ReadyPath $ReadyFile.FullName
    if ($generation -lt $MinimumGeneration) {
        return $false
    }

    $stageDir = $ReadyFile.DirectoryName
    $archivePath = Join-Path $stageDir "submission.tar.gz"
    $messagePath = Join-Path $stageDir "submission.message.txt"
    $submittedPath = Join-Path $stageDir "submission.submitted"
    $failedPath = Join-Path $stageDir "submission.failed"
    $submittingPath = Join-Path $stageDir "submission.submitting"

    if ((Test-Path $submittedPath) -or (Test-Path $failedPath)) {
        return $false
    }
    if (!(Test-Path $archivePath)) {
        Write-SubmitLog "archive_missing ready=$($ReadyFile.FullName) archive=$archivePath"
        return $false
    }
    if (!(Test-Path $messagePath)) {
        Write-SubmitLog "message_missing ready=$($ReadyFile.FullName) message=$messagePath"
        return $false
    }

    try {
        $lock = New-Item -Path $submittingPath -ItemType File -ErrorAction Stop
    } catch {
        return $false
    }

    $message = (Get-Content -Raw -Path $messagePath).Trim()
    Write-SubmitLog "submit_start generation=$generation archive=$archivePath message=$message"
    $output = & $KaggleCommand competitions submit -c $Competition -f $archivePath -m $message 2>&1
    $exitCode = $LASTEXITCODE
    $outputText = ($output | Out-String).Trim()

    if ($exitCode -eq 0) {
        Set-Content -Path $submittedPath -Value "submitted_at=$((Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ"))`n$outputText"
        Remove-Item -LiteralPath $submittingPath -Force
        Write-SubmitLog "submit_done generation=$generation archive=$archivePath output=$outputText"
        return $true
    }

    Set-Content -Path $failedPath -Value "failed_at=$((Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ"))`nexit_code=$exitCode`n$outputText"
    Remove-Item -LiteralPath $submittingPath -Force
    Write-SubmitLog "submit_failed generation=$generation exit_code=$exitCode archive=$archivePath output=$outputText"
    return $false
}

Write-SubmitLog "watcher_start stage_root=$StageRoot competition=$Competition minimum_generation=$MinimumGeneration"

while ($true) {
    if (Test-Path $StageRoot) {
        $readyFiles = Get-ChildItem -Path $StageRoot -Recurse -Filter "submission.ready" -File |
            Sort-Object FullName
        foreach ($readyFile in $readyFiles) {
            $submitted = Submit-ReadyFile -ReadyFile $readyFile
            if ($submitted -and $ExitAfterFirstSubmit) {
                Write-SubmitLog "watcher_exit_after_first_submit"
                exit 0
            }
        }
    }
    Start-Sleep -Seconds $PollSeconds
}
