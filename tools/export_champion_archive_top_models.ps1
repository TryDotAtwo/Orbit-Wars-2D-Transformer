param(
    [string]$CheckpointPath = "artifacts/full_training_2026-06-01_self_train_current_contract.checkpoint.bin",
    [string]$OutputRoot = "dashboard/public/telemetry/generation_models",
    [switch]$Overwrite
)

$CheckpointMagic = "OWTRAIN1"
$TopModelsMagic = "OWTOP4_1"
$ExpectedTopModelsPerGeneration = 4
$BytesPerFloat = 4

function Read-RequiredBytes {
    param(
        [System.IO.BinaryReader]$Reader,
        [int]$Count,
        [string]$Name
    )
    $bytes = $Reader.ReadBytes($Count)
    if ($bytes.Length -ne $Count) {
        throw "short_read=$Name; expected=$Count; actual=$($bytes.Length)"
    }
    return ,$bytes
}

function Read-U64 {
    param([System.IO.BinaryReader]$Reader, [string]$Name)
    $bytes = Read-RequiredBytes -Reader $Reader -Count 8 -Name $Name
    return [System.BitConverter]::ToUInt64($bytes, 0)
}

function Read-U32 {
    param([System.IO.BinaryReader]$Reader, [string]$Name)
    $bytes = Read-RequiredBytes -Reader $Reader -Count 4 -Name $Name
    return [System.BitConverter]::ToUInt32($bytes, 0)
}

function Write-U64 {
    param([System.IO.BinaryWriter]$Writer, [UInt64]$Value)
    $Writer.Write([System.BitConverter]::GetBytes($Value))
}

function Write-I32 {
    param([System.IO.BinaryWriter]$Writer, [int]$Value)
    $Writer.Write([System.BitConverter]::GetBytes([int]$Value))
}

function Read-TransformerBlockBytes {
    param([System.IO.BinaryReader]$Reader)
    $buffer = New-Object System.IO.MemoryStream
    $writer = [System.IO.BinaryWriter]::new($buffer)
    try {
        foreach ($field in @("model_layers", "model_row_count", "model_weight_count")) {
            $bytes = Read-RequiredBytes -Reader $Reader -Count 8 -Name $field
            $writer.Write($bytes)
            if ($field -eq "model_weight_count") {
                $weightCount = [System.BitConverter]::ToUInt64($bytes, 0)
            }
        }
        $weightBytes = [int64]$weightCount * $BytesPerFloat
        if ($weightBytes -gt [int]::MaxValue) {
            throw "model_weight_bytes_too_large=$weightBytes"
        }
        $weights = Read-RequiredBytes -Reader $Reader -Count ([int]$weightBytes) -Name "model_weights"
        $writer.Write($weights)
        return ,$buffer.ToArray()
    } finally {
        $writer.Dispose()
        $buffer.Dispose()
    }
}

function Skip-TransformerBlock {
    param([System.IO.BinaryReader]$Reader)
    [void](Read-U64 -Reader $Reader -Name "model_layers")
    [void](Read-U64 -Reader $Reader -Name "model_row_count")
    $weightCount = [int64](Read-U64 -Reader $Reader -Name "model_weight_count")
    $weightBytes = $weightCount * $BytesPerFloat
    $newPosition = $Reader.BaseStream.Seek($weightBytes, [System.IO.SeekOrigin]::Current)
    if ($newPosition -gt $Reader.BaseStream.Length) {
        throw "skip_past_end=model_weights; requested=$weightBytes"
    }
}

$resolvedCheckpoint = Resolve-Path -LiteralPath $CheckpointPath
$workspace = (Resolve-Path -LiteralPath ".").Path
$resolvedOutputRoot = Join-Path $workspace $OutputRoot
New-Item -ItemType Directory -Path $resolvedOutputRoot -Force | Out-Null

$stream = [System.IO.File]::Open($resolvedCheckpoint.Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
try {
    $reader = [System.IO.BinaryReader]::new($stream)
    $magic = [System.Text.Encoding]::ASCII.GetString((Read-RequiredBytes -Reader $reader -Count $CheckpointMagic.Length -Name "checkpoint_magic"))
    if ($magic -ne $CheckpointMagic) {
        throw "checkpoint_magic_mismatch=$magic"
    }
    $version = Read-U32 -Reader $reader -Name "checkpoint_version"
    if ($version -ne 1) {
        throw "checkpoint_version_unsupported=$version"
    }
    $runIdLength = [int](Read-U64 -Reader $reader -Name "run_id_length")
    $runId = [System.Text.Encoding]::UTF8.GetString((Read-RequiredBytes -Reader $reader -Count $runIdLength -Name "run_id"))
    $nextGeneration = Read-U64 -Reader $reader -Name "next_generation"
    $populationCount = [int](Read-U64 -Reader $reader -Name "population_count")
    for ($index = 0; $index -lt $populationCount; $index++) {
        Skip-TransformerBlock -Reader $reader
    }

    $championCount = [int](Read-U64 -Reader $reader -Name "champion_count")
    $championsByGeneration = [ordered]@{}
    for ($index = 0; $index -lt $championCount; $index++) {
        $generation = [int](Read-U64 -Reader $reader -Name "champion_generation")
        $generationKey = [string]$generation
        $modelBytes = Read-TransformerBlockBytes -Reader $reader
        if (-not $championsByGeneration.Contains($generationKey)) {
            $championsByGeneration[$generationKey] = New-Object System.Collections.Generic.List[byte[]]
        }
        $championsByGeneration[$generationKey].Add($modelBytes)
    }
    if ($stream.Position -ne $stream.Length) {
        throw "checkpoint_trailing_bytes=$($stream.Length - $stream.Position)"
    }

    $runOutputDir = Join-Path $resolvedOutputRoot $runId
    New-Item -ItemType Directory -Path $runOutputDir -Force | Out-Null

    $written = 0
    $skipped = 0
    $rows = @()
    foreach ($generationKey in ($championsByGeneration.Keys | Sort-Object {[int]$_})) {
        $generation = [int]$generationKey
        $models = $championsByGeneration[$generationKey]
        if ($models.Count -ne $ExpectedTopModelsPerGeneration) {
            throw "champion_generation_count_mismatch=generation_$generation; count=$($models.Count); expected=$ExpectedTopModelsPerGeneration"
        }
        $targetPath = Join-Path $runOutputDir ("generation_{0}_top4.owmodels" -f $generation)
        if ((Test-Path -LiteralPath $targetPath) -and -not $Overwrite) {
            $skipped += 1
            $rows += [pscustomobject]@{ Generation = $generation; Action = "skipped_existing"; Path = $targetPath }
            continue
        }

        $output = [System.IO.File]::Open($targetPath, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
        try {
            $writer = [System.IO.BinaryWriter]::new($output)
            $writer.Write([System.Text.Encoding]::ASCII.GetBytes($TopModelsMagic))
            Write-U64 -Writer $writer -Value ([UInt64]$generation)
            Write-U64 -Writer $writer -Value ([UInt64]$ExpectedTopModelsPerGeneration)
            for ($modelIndex = 0; $modelIndex -lt $ExpectedTopModelsPerGeneration; $modelIndex++) {
                Write-U64 -Writer $writer -Value ([UInt64]$modelIndex)
                Write-I32 -Writer $writer -Value 0
                $modelBytes = [byte[]]$models[$modelIndex]
                $writer.BaseStream.Write($modelBytes, 0, $modelBytes.Length)
            }
        } finally {
            if ($writer) {
                $writer.Dispose()
            }
            $output.Dispose()
        }
        $written += 1
        $rows += [pscustomobject]@{ Generation = $generation; Action = "written"; Path = $targetPath }
    }

    [pscustomobject]@{
        RunId = $runId
        NextGeneration = $nextGeneration
        PopulationCount = $populationCount
        ChampionCount = $championCount
        Generations = $championsByGeneration.Count
        Written = $written
        Skipped = $skipped
    } | Format-List
    $rows | Format-Table -AutoSize
} finally {
    if ($reader) {
        $reader.Dispose()
    }
    $stream.Dispose()
}
