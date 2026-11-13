# Reproduce the experiment

## Software

Use a working [MotrixLab checkout](https://github.com/Motphys/MotrixLab/tree/24796cff1c00591bd327378df93461e1c4a9f89c) at commit `24796cff1c00591bd327378df93461e1c4a9f89c`. The runs used Windows, Python 3.10.21, MotrixLab 0.4.0, MotrixSim 0.10.1, skrl 2.1.0 and PyTorch 2.7.0+cu128 on an RTX 5080 Laptop GPU. Follow the upstream setup for that checkout and ensure the Go2 Git LFS meshes are present. The project scripts use private simulator hooks tied to these versions.

## Train

Adjust these two paths if needed. Run from the MotrixLab checkout with its virtual environment.

```powershell
$motrix = "$HOME\Projects\MotrixLab"
$study = "$HOME\Projects\Go2-Robust-RL"
Set-Location -LiteralPath $motrix
$python = Join-Path $motrix '.venv\Scripts\python.exe'
$batch = Join-Path $study ('runs-audit\' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
foreach ($group in @('A', 'B', 'C', 'D')) {
    foreach ($seed in @(7, 17, 27)) {
        $audit = Join-Path $batch "${group}_seed${seed}"
        & $python "$study\scripts\train_project.py" --group $group --steps 60000 --seed $seed --num-envs 2048 --audit-dir $audit
        if ($LASTEXITCODE -ne 0) { throw "Training failed for $group seed $seed" }
    }
}
```

Each audit's `experiment.json` records the run location. Preserve the checkpoint and the surrounding run directory so MotrixLab can recover the policy metadata and configuration.

## Evaluate

Still in the MotrixLab checkout, select one completed audit folder and use its group and policy path.

```powershell
$audit = Join-Path $batch 'A_seed7'
$record = Get-Content -LiteralPath (Join-Path $audit 'experiment.json') -Raw | ConvertFrom-Json
$policy = Join-Path $record.run_dir_local 'checkpoints\latest.pt'
foreach ($mode in @('nominal', 'push', 'timing')) {
    & $python "$study\scripts\evaluate_$mode.py" --group $record.group --policy $policy --output (Join-Path $audit $mode)
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed for $mode" }
}
```

For an original checkpoint, add `--training-seed 7`, `17` or `27` to verify its SHA-256 against the training audit. Omit this option for a newly trained policy. Original checkpoint binaries remain on the training machine.

## Recompute published results

From the project checkout, use Python 3.10 or later. The summary and tests use the standard library. Figures require Matplotlib and MP4 generation also requires FFmpeg on PATH.

```text
python -m unittest discover -s scripts
python scripts/summarize.py
python scripts/summarize.py --plot
python scripts/animate.py
```

The episode table retains command, push amplitude and onset, first fall, recovery latency and planar tracking error. Blank fields indicate an inapplicable condition or an event that did not occur. The importer reconstructs this table and verifies the force schedule from the original study archives when supplied as a directory argument. The two example CSVs are the full control traces for the selected matched comparison.

## Record the matched 3D comparison

The two uploaded seed-7 model ZIPs have been checked against the original checkpoint hashes. Place them in Downloads as `Go2_A_seed7_checkpoint.zip` and `Go2_D_seed7_checkpoint.zip`. From a local copy of this repository, run the launcher once per group.

```powershell
powershell -ExecutionPolicy Bypass -File .\Run_Replay.ps1 -Group A
powershell -ExecutionPolicy Bypass -File .\Run_Replay.ps1 -Group D
```

The launcher downloads the replay dependencies, extracts the model into the expected run layout and verifies its hash. The viewer pauses at the initial pose. Start a Windows screen recording of MotrixRender, then press Enter in the terminal. The rollout uses the original deterministic action selection, evaluation seed 101 and +60 N pulse at 3.0 s for 0.20 s. It presents simulator states before automatic reset, including the first fall frame, and pauses at completion for recording to finish. Both runs start with the same camera settings. Each run also writes fresh CSV telemetry to its printed output folder for comparison with the original data.

The viewer addition has passed Python compilation and source checks. GPU rendering and the PowerShell launcher require validation on the training Windows machine.
