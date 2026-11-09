"""A pilot nominal diagnostic evaluation for the uploaded MotrixLab 0.4.0 Go2 task.
Uses the installed CLI to restore the exact saved policy and preprocessors.
No source files are edited. Private task hooks are version-sensitive.
"""
import argparse
from audit_policy import verify_policy
import csv
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import numpy as np

SCENARIOS = [(f'forward_{v:.1f}', (v, 0., 0.)) for v in (0., .2, .5, .8, 1., 1.2)] + [
    ('turn_left', (.5, 0., .5)), ('turn_right', (.5, 0., -.5)),
    ('side_left', (0., .3, 0.)), ('side_right', (0., -.3, 0.))]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


class Metrics:
    def __init__(self, warmup):
        self.warmup = warmup
        self.fallen = False
        self.first_fall_s = None
        self.samples = []
        self.low_count = 0
        self.observed = 0

    def update(self, t, vx, vy, wz, height, up_z, command, terminated):
        values = np.array([t, vx, vy, wz, height, up_z, *command], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError('Non-finite simulator values: evaluation stopped')
        # Only the initial trajectory is scored; auto-reset trajectories are ignored.
        if self.fallen:
            return
        self.observed += 1
        self.low_count += int(height < .20)
        if terminated:
            self.fallen = True
            self.first_fall_s = t
            return
        if t > self.warmup:
            dx, dy, dw = vx-command[0], vy-command[1], wz-command[2]
            self.samples.append((dx, dy, dw))

    def summary(self):
        out = {'fallen': self.fallen, 'first_fall_s': self.first_fall_s,
               'scored_steps': len(self.samples), 'observed_steps_until_first_fall': self.observed,
               'low_height_fraction_until_first_fall': self.low_count/self.observed if self.observed else None}
        if self.samples:
            a = np.asarray(self.samples)
            out.update(planar_rmse_m_s=float(np.sqrt(np.mean(np.sum(a[:, :2]**2, axis=1)))),
                       mean_planar_error_m_s=float(np.mean(np.linalg.norm(a[:, :2], axis=1))),
                       forward_mae_m_s=float(np.mean(np.abs(a[:, 0]))),
                       yaw_rmse_rad_s=float(np.sqrt(np.mean(a[:, 2]**2))))
        else:
            out.update(planar_rmse_m_s=None, mean_planar_error_m_s=None,
                       forward_mae_m_s=None, yaw_rmse_rad_s=None)
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--quick', action='store_true', help='One 0.5 m/s scenario; interface smoke test only')
    parser.add_argument('--group', choices=['A', 'B', 'C', 'D'], required=True)
    parser.add_argument('--training-seed', type=int, choices=(7, 17, 27),
                        help='Verify an original study checkpoint against the published audit')
    args = parser.parse_args()
    policy = args.policy.resolve()
    if not policy.is_file():
        parser.error('Checkpoint does not exist')
    if args.output.exists():
        parser.error('Output directory exists; choose a new output name')
    args.output.mkdir(parents=True)

    verify_policy(policy, args.group, args.training_seed)
    # Saved task_config contains the custom task name, not an environment snapshot.
    # For nominal evaluation, the same Go2 task uses upstream for_play() settings:
    # no dynamics randomization, no observation noise, and fixed commands below.
    from motrix_env_core import registry
    from motrix_envs.locomotion.quadruped.go2 import Go2WalkDirectEnvCfg
    from motrix_envs.locomotion.quadruped.walk_np import QuadrupedWalkTask
    if not registry.contains('go2-walk-flat'):
        raise RuntimeError('Upstream Go2 task is unavailable')
    env_id = 'go2-project-' + args.group.lower()
    if registry.contains(env_id):
        raise RuntimeError('Unexpected existing project registration; use a fresh Python process')
    registry.envcfg(env_id, cfg_type=Go2WalkDirectEnvCfg)(lambda: Go2WalkDirectEnvCfg())
    registry.env(env_id)(QuadrupedWalkTask)
    import torch
    from motrix_rl_skrl.torch.train.ppo import Trainer
    from motrix_rl_skrl.ppo import add_runtime_config
    from motrix_rl.config import LoggingConfig, CheckpointConfig

    forces = {f"push_{f:+d}N": f for f in (0, 40, -40, 60, -60, 80, -80)}
    scenarios = [(name, (.5, 0., 0.)) for name in forces]
    report = {'protocol': 'push-paired-pilot-v1', 'group': args.group, 'training_command_upper_vx': 1.2, 'status': 'running',
              'policy_sha256': sha256(policy), 'eval_seed': args.seed,
              'quick_interface_test': args.quick,
              'scope': 'One rollout per signed force at one command and push onset; exploratory comparison.',
              'push_start_s': 3., 'requested_window_s': 0.2, 'warmup_seconds': 2., 'duration_seconds': 8.,
              'failure_rule': 'Upstream termination: up_z <= configured termination_min_up_z; no base-height termination.',
              'tracking_rule': 'After warmup, before first termination; failures reported separately, auto-reset rollouts excluded.',
              'low_height_rule': 'World base height below 0.20 m on flat ground; diagnostic only, not failure.',
              'packages': {}, 'source_hashes': {}, 'scenarios': []}
    for name in ['motrixlab', 'motrixsim', 'motrix-envs', 'motrix-rl-skrl', 'torch', 'skrl']:
        report['packages'][name] = metadata.version(name)
    import motrix_envs.locomotion.quadruped.walk_np as walk_module
    import motrix_envs.locomotion.quadruped.go2 as go2_module
    for label, module in [('task', walk_module), ('go2', go2_module)]:
        report['source_hashes'][label] = sha256(module.__file__)

    def save():
        (args.output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')

    # Keep restoration under MotrixLab's installed CLI; replace only its play loop in memory.
    def diagnostic_play(self, checkpoint):
        for index, (name, cmd) in enumerate(scenarios):
            print(f'[{index+1}/{len(scenarios)}] {name}', flush=True)
            self._set_seed(args.seed)
            raw = self._make_env(num_envs=1, mode='play')
            env = None
            try:
                if not all(hasattr(raw, k) for k in ['resample_commands', 'compute_transition',
                    'get_local_linvel', 'get_gyro', 'get_gravity', 'sim_data']):
                    raise RuntimeError('Unexpected environment API; send error_log.txt back for compatibility repair')
                if raw.cfg.randomization.enabled or raw.cfg.noise_config.level != 0:
                    raise RuntimeError('Nominal evaluation requires disabled randomization and observation noise')
                dt = float(raw.cfg.ctrl_dt)
                duration = report['duration_seconds']
                steps = round(duration/dt)
                tracker = Metrics(warmup=2.)
                capture = {}
                sim = raw.sim
                model, data = sim._model, sim._data
                link = model.get_link(raw._base_link_name)
                if link is None:
                    raise RuntimeError(f'Cannot resolve base link {raw._base_link_name!r}')
                substeps = int(raw.cfg.sim_substeps)
                sim_dt = dt / substeps
                pulse_start = round(3. / sim_dt)
                pulse_steps = round(.2 / sim_dt)
                if abs(pulse_steps * sim_dt - .2) > 1e-9:
                    raise RuntimeError('0.2 s is not an integer number of physics steps')
                force = np.zeros((*data.shape, 3), dtype=np.float32)
                force[..., 1] = forces[name]
                physics_index = 0
                calls = []
                physics_trace = []
                def calibration_step():
                    nonlocal physics_index
                    for _ in range(substeps):
                        active = (
                            forces[name] != 0
                            and pulse_start <= physics_index < pulse_start + pulse_steps
                        )
                        if active:
                            link.add_external_force(data, force, None, False)
                            calls.append(physics_index)
                        sim.step(1)
                        physics_trace.append((physics_index, physics_index * sim_dt,
                                              (physics_index + 1) * sim_dt, forces[name] if active else 0.))
                        physics_index += 1
                raw.physics_step = calibration_step
                command = np.asarray(cmd, dtype=np.float32)
                raw.resample_commands = lambda n: np.tile(command, (n, 1))
                original_transition = raw.compute_transition

                def measured_transition(state):
                    result = original_transition(state)
                    capture['v'] = np.asarray(raw.get_local_linvel()).copy()
                    capture['w'] = np.asarray(raw.get_gyro()).copy()
                    capture['up'] = np.asarray(raw.get_gravity()).copy()
                    capture['pos'] = np.asarray(raw.sim_data['base_pos']).copy()
                    capture['terminated'] = np.asarray(result.terminated).copy()
                    return result

                raw.compute_transition = measured_transition
                env = self._wrap_env(raw, None)
                models = self._make_model(env, self._rlcfg)
                ppo_cfg = self._rlcfg.agent.to_dict()
                add_runtime_config(ppo_cfg, env, scheduler_cls=type(self).scheduler_cls,
                    scaler_cls=type(self).scaler_cls, logging=LoggingConfig(backend='tensorboard'), checkpoint=CheckpointConfig())
                agent = self._make_agent(models, env, ppo_cfg, self._rlcfg.memory)
                agent.load(checkpoint)
                # Eval mode prevents running-stat updates; mean_actions removes action sampling.
                agent.enable_training_mode(False, apply_to_models=True)
                for model in models.values():
                    model.eval()
                obs, _ = env.reset()
                state = env.state()
                with (args.output / (name + '.csv')).open('w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['step','time_s','cmd_vx','cmd_vy','cmd_wz','vx','vy','wz',
                                     'height_m','up_z','terminated','first_trajectory'])
                    with torch.no_grad():
                        for step in range(steps):
                            capture.clear()
                            actions, outputs = agent.act(obs, state, timestep=0, timesteps=0)
                            if 'mean_actions' not in outputs:
                                raise RuntimeError('Policy did not expose mean_actions; deterministic evaluation unavailable')
                            obs, _, _, _, _ = env.step(outputs['mean_actions'])
                            state = env.state()
                            if physics_index != (step + 1) * substeps:
                                raise RuntimeError('Calibration hook did not run the expected physics substeps')
                            if not capture:
                                raise RuntimeError('Transition hook was not called; cannot measure pre-reset states')
                            v, w, p, up = capture['v'][0], capture['w'][0], capture['pos'][0], capture['up'][0]
                            done = bool(capture['terminated'].reshape(-1)[0])
                            t = (step+1)*dt
                            initial = not tracker.fallen
                            tracker.update(t, float(v[0]), float(v[1]), float(w[2]), float(p[2]),
                                           float(up[2]), command, done)
                            writer.writerow([step+1,t,*cmd,float(v[0]),float(v[1]),float(w[2]),
                                             float(p[2]),float(up[2]),done,initial])
                            if tracker.fallen:
                                break
                row = {'name': name, 'command': list(cmd), 'ctrl_dt_s': dt,
                       'out_of_training_forward_range': cmd[0] > 1.2,
                       'termination_min_up_z': float(raw.cfg.termination_min_up_z), **tracker.summary()}
                with (args.output / (name + '_physics.csv')).open('w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['physics_step', 'start_s', 'end_s', 'force_added_y_N'])
                    writer.writerows(physics_trace)
                row.update(physics_dt_s=sim_dt, physics_substeps=substeps,
                           base_link=raw._base_link_name, add_force_call_steps=calls,
                           intended_repeat_substeps=pulse_steps,
                           requested_force_world_y_N=forces[name],
                           pulse_completed=(len(calls) == pulse_steps) if forces[name] else None)
                report['scenarios'].append(row)
                save()
                print(json.dumps(row), flush=True)
            finally:
                if env is not None:
                    env.close()
                else:
                    raw.close()
        report['status'] = 'completed'
        save()
        print('Evaluation complete:', args.output.resolve(), flush=True)

    Trainer.play = diagnostic_play
    save()
    entry = next(ep for ep in metadata.entry_points(group='console_scripts') if ep.name == 'mxlab')
    sys.argv = ['mxlab', 'play', 'policy=' + json.dumps(policy.as_posix()), 'num_envs=1']
    try:
        result = entry.load()()
        if report['status'] != 'completed':
            raise RuntimeError(f'CLI exited without completing evaluation: {result}')
    except BaseException as e:
        if report['status'] != 'completed':
            report['status'] = 'failed'
            report['error'] = str(e)
            save()
        raise

if __name__ == '__main__':
    main()
