"""Optional checkpoint provenance check for the twelve published study runs."""
import hashlib
import json
from pathlib import Path


def verify_policy(policy, group, training_seed):
    if training_seed is None:
        return
    manifest = Path(__file__).resolve().parents[1] / 'data' / 'training_runs.json'
    runs = json.loads(manifest.read_text(encoding='utf-8'))
    expected = next(run['checkpoint_sha256'] for run in runs
                    if run['group'] == group and run['seed'] == training_seed)
    actual = hashlib.sha256(policy.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f'Checkpoint does not match group {group}, seed {training_seed}: {actual}')
