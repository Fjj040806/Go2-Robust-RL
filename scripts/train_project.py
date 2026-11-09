"""Register isolated project environments and invoke the installed MotrixLab CLI.
Does not edit upstream files. Re-run via this launcher to register custom envs.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import sys
import traceback

from experiment_spec import GROUPS, LOWER, UPPER, curriculum_scale, group_spec


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', choices=list(GROUPS), required=True)
    parser.add_argument('--steps', type=int, default=240)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--num-envs', type=int, default=2048)
    parser.add_argument('--audit-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 120 or args.steps % 24 or args.num_envs < 4:
        parser.error('Steps must be a multiple of 24 and at least 120; num-envs must be at least 4')
    audit = args.audit_dir.resolve()
    if audit.exists():
        parser.error('Audit directory already exists; use a new directory')
    audit.mkdir(parents=True)
    flags = GROUPS[args.group]
    spec = group_spec(args.group, args.steps, args.seed, args.num_envs)
    spec['status'] = 'initializing'
    spec['source_hashes'] = {}
    spec['packages'] = {name: metadata.version(name) for name in
                        ['motrixlab','motrixsim','motrix-envs','motrix-rl-skrl','skrl','torch']}
    spec['launcher_sha256'] = digest(__file__)
    spec['spec_sha256'] = digest(Path(__file__).with_name('experiment_spec.py'))

    def write_json(path, value):
        def convert(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            if isinstance(obj, Path):
                return str(obj)
            return repr(obj)
        path.write_text(json.dumps(value, indent=2, default=convert, allow_nan=False), encoding='utf-8')

    def save():
        write_json(audit / 'experiment.json', spec)

    save()
    import numpy as np
    from motrix_env_core import registry
    from motrix_env_core.config import configclass
    from motrix_envs.locomotion.quadruped.go2 import Go2WalkDirectEnvCfg
    from motrix_envs.locomotion.quadruped.cfg import QuadrupedWalkRandomizationCfg
    from motrix_envs.locomotion.quadruped.walk_np import QuadrupedWalkTask
    from motrix_rl_skrl.torch.train.ppo import Trainer
    import motrix_envs.locomotion.quadruped.walk_np as upstream_task
    import motrix_envs.locomotion.quadruped.go2 as upstream_go2
    for label, module in [('task',upstream_task),('go2',upstream_go2)]:
        spec['source_hashes'][label] = digest(module.__file__)
    # Discover upstream providers before registering local environments.
    if not registry.contains('go2-walk-flat'):
        raise RuntimeError('Upstream Go2 task is not registered')

    @configclass
    class ProjectCfg(Go2WalkDirectEnvCfg):
        project_curriculum: bool = False
        project_steps: int = 60000

        def for_play(self):
            return replace(super().for_play(), project_curriculum=False)

    def factory():
        base = ProjectCfg()
        velocity = replace(base.commands.velocity,
                           lower=np.asarray(LOWER,dtype=np.float32),
                           upper=np.asarray(UPPER,dtype=np.float32),
                           standing_probability=0.1, resampling_seconds_range=(5.,10.))
        randomization = QuadrupedWalkRandomizationCfg(
            enabled=flags['randomization'],
            sliding_friction_range=(0.4,0.9) if flags['randomization'] else None,
            base_mass_scale_range=(0.9,1.1) if flags['randomization'] else (1.,1.),
            kp_scale_range=(0.9,1.1) if flags['randomization'] else (1.,1.),
            damping_scale_range=(0.9,1.1) if flags['randomization'] else (1.,1.))
        return replace(base, project_curriculum=flags['curriculum'], project_steps=args.steps,
                       commands=replace(base.commands,velocity=velocity), randomization=randomization)

    class ProjectTask(QuadrupedWalkTask):
        def __init__(self, cfg, num_envs=1, backend=None):
            self._project_step = 0
            self._project_scale = curriculum_scale(0,cfg.project_steps,cfg.project_curriculum)
            self._project_audited = False
            super().__init__(cfg,num_envs=num_envs,backend=backend)
            write_json(audit/'effective_env_config.json',asdict(cfg))
            actual = {'kp':self._randomize_kp,'damping':self._randomize_damping,
                      'friction':self._randomize_friction,'mass':self._randomize_base_mass}
            assert all(bool(v)==flags['randomization'] for v in actual.values()), actual
            assert not self._randomize_action_delay and not self._randomize_base_com
            spec['runtime_randomization_switches'] = actual
            spec['num_actions'] = int(self._num_action)
            spec['policy_obs_dim'] = int(self._policy_obs_dim())
            spec['value_obs_dim'] = int(self._value_obs_dim())
            spec['ctrl_dt_s'] = float(cfg.ctrl_dt)
            save()
            self._log_stage()

        def _log_stage(self):
            event = {'step':self._project_step,'scale':self._project_scale,
                     'lower':(np.asarray(LOWER)*self._project_scale).tolist(),
                     'upper':(np.asarray(UPPER)*self._project_scale).tolist()}
            with (audit/'curriculum.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(event)+'\n')
            print('PROJECT command range:',json.dumps(event),flush=True)

        def resample_commands(self,num_envs):
            return super().resample_commands(num_envs)*self._project_scale

        def compute_transition(self,state):
            scale = curriculum_scale(self._project_step,self.cfg.project_steps,self.cfg.project_curriculum)
            if scale != self._project_scale:
                # Expand current commands too; do not wait for another resample.
                self._commands *= scale/self._project_scale
                self._project_scale = scale
                self._log_stage()
            self._project_step += 1
            spec["observed_environment_control_steps"] = self._project_step
            return super().compute_transition(state)

        def _randomize_params(self,env_ids):
            super()._randomize_params(env_ids)
            if not self._project_audited:
                result = {'num_reset':len(env_ids),'randomization_enabled':bool(self.cfg.randomization.enabled)}
                if self._randomize_writes is not None:
                    for key in ['kp','damping','friction','mass']:
                        a=np.asarray(self._randomize_writes.buffer(key)[env_ids])
                        if not np.isfinite(a).all():
                            raise RuntimeError('Non-finite randomization buffer: '+key)
                        result[key]={'min':float(a.min()),'max':float(a.max()),'std':float(a.std()),'shape':list(a.shape)}
                write_json(audit/'first_reset_randomization.json',result)
                self._project_audited=True

    env_id='go2-project-'+args.group.lower()
    registry.envcfg(env_id,cfg_type=ProjectCfg)(factory)
    registry.env(env_id)(ProjectTask)
    original_train=Trainer.train

    def tracked_train(self):
        run_dir=Path(self._context.run_dir)
        spec['run_dir_local']=str(run_dir.resolve())
        spec['status']='training'
        save()
        original_train(self)
        policy=run_dir/'checkpoints'/'latest.pt'
        if not policy.is_file():
            raise RuntimeError('Training returned without latest.pt')
        spec['checkpoint_sha256']=digest(policy)
        if spec.get('observed_environment_control_steps') != args.steps:
            raise RuntimeError('Observed environment steps do not match requested trainer steps')
        shutil.copy2(run_dir/'task_config.yaml',audit/'task_config.yaml')
        # Preserve local custom environment definition beside the saved run.
        for name in ['train_project.py','experiment_spec.py']:
            shutil.copy2(Path(__file__).with_name(name),run_dir/name)
        spec['status']='completed'
        save()
        shutil.copy2(audit/'experiment.json',run_dir/'project_experiment.json')
        print('PROJECT TRAINING COMPLETED:',args.group,flush=True)

    Trainer.train=tracked_train
    checkpoint_interval=240 if args.steps==240 else 1000
    sys.argv=['mxlab','train','task=go2-walk-flat/skrl.ppo.torch','task.env='+env_id,
              'num_envs='+str(args.num_envs),'seed='+str(args.seed),
              'algo.trainer.timesteps='+str(args.steps),'logging.interval=24',
              'checkpoint.interval='+str(checkpoint_interval),'render=false','play=false']
    entry=next(ep for ep in metadata.entry_points(group='console_scripts') if ep.name=='mxlab')
    try:
        entry.load()()
        if spec['status']!='completed':
            raise RuntimeError('CLI returned before completing project training')
    except BaseException:
        if spec['status']!='completed':
            spec['status']='failed'
            save()
            (audit/'error_log.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise

if __name__=='__main__':main()
