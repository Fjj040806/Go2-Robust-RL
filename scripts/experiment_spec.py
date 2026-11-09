"""Pure experiment definitions; shared by training and tests."""
GROUPS = {
    'A': {'curriculum': False, 'randomization': False},
    'B': {'curriculum': True, 'randomization': False},
    'C': {'curriculum': False, 'randomization': True},
    'D': {'curriculum': True, 'randomization': True},
}
LOWER = (-0.5, -0.5, -0.5)
UPPER = (1.2, 0.5, 0.5)
STAGES = ((0.0, 0.25), (0.2, 0.5), (0.5, 0.75), (0.8, 1.0))


def curriculum_scale(step, total_steps, enabled):
    if total_steps <= 0 or step < 0:
        raise ValueError('Invalid training progress')
    if not enabled:
        return 1.0
    progress = step / total_steps
    return max((scale for start, scale in STAGES if progress >= start), default=0.25)


def group_spec(group, total_steps, seed, num_envs):
    return {'protocol': 'go2-factorial-v1-pilot', 'group': group,
            **GROUPS[group], 'trainer_steps': total_steps, 'seed': seed, 'num_envs': num_envs,
            'command_lower': list(LOWER), 'command_upper': list(UPPER),
            'curriculum_stages_progress_scale': list(STAGES),
            'curriculum_type': 'Time-scheduled command-range expansion; not performance-adaptive.',
            'randomization_scope': 'Only friction, base-link mass, actuator Kp and damping.',
            'shared_observation_noise': 'Upstream NoiseConfig unchanged in all four groups.',
            'standing_probability': 0.1, 'command_resampling_seconds': [5., 10.],
            'reward': 'Unchanged upstream Go2 reward.',
            'evaluation_status': 'Not evaluated. No push-recovery claims.'}
