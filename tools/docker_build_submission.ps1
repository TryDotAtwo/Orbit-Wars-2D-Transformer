$ErrorActionPreference = "Stop"

$workspace = (Get-Location).Path
docker run --rm -v "${workspace}:/workspace" -w /workspace cmz-native-dev:2026-05-26 bash tools/build_submission.sh

