param(
    [ValidateSet('A', 'D')][string]$Group = 'A',
    [string]$MotrixRoot = "$HOME\Projects\MotrixLab",
    [string]$CheckpointZip = ""
)
$ErrorActionPreference = 'Stop'
if (-not $CheckpointZip) {
    $CheckpointZip = "$HOME\Downloads\Go2_${Group}_seed7_checkpoint.zip"
}
$python = Join-Path $MotrixRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw "Python not found at $python" }
if (-not (Test-Path -LiteralPath $CheckpointZip)) { throw "Checkpoint ZIP not found at $CheckpointZip" }
$work = Join-Path "$HOME\Downloads" ("Go2_Replay_${Group}_" + (Get-Date -Format 'yyyyMMdd_HHmmss'))
$run = Join-Path $work 'model'
New-Item -ItemType Directory -Path (Join-Path $work 'scripts'), (Join-Path $work 'data'), (Join-Path $run 'checkpoints') -Force | Out-Null
$repo = 'https://raw.githubusercontent.com/Fjj040806/Go2-Robust-RL/main'
foreach ($file in @('scripts/evaluate_push.py', 'scripts/audit_policy.py', 'data/training_runs.json')) {
    Invoke-WebRequest -UseBasicParsing -Uri "$repo/$file" -OutFile (Join-Path $work $file)
}
Expand-Archive -LiteralPath $CheckpointZip -DestinationPath $run
Move-Item -LiteralPath (Join-Path $run 'latest.pt') -Destination (Join-Path $run 'checkpoints\latest.pt')
Push-Location -LiteralPath $MotrixRoot
try {
    & $python (Join-Path $work 'scripts\evaluate_push.py') --group $Group --policy (Join-Path $run 'checkpoints\latest.pt') --training-seed 7 --viewer --output (Join-Path $work 'results')
    if ($LASTEXITCODE -ne 0) { throw "Replay failed. Keep the terminal output for diagnosis." }
} finally {
    Pop-Location
}
Write-Host "Replay and measured telemetry saved in $work"
