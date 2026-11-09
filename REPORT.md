# Command curriculum and dynamics randomization for Unitree Go2

This simulation study measures how two training interventions affect velocity tracking and lateral-push recovery on flat ground. The experiment crosses command curriculum and dynamics randomization in a 2 × 2 design, with training seeds 7, 17 and 27 in each condition. The fixed-condition baseline performed best on the reported tracking and push metrics under this protocol.

## Training design

The policies share the upstream Go2 reward and observation noise. Each run uses 2,048 parallel environments for 60,000 vectorized control steps, yielding 122.88 million transitions per run and 1.475 billion across the twelve runs. PPO uses 24 rollout steps per update, four minibatches and five learning epochs, with 512/256/128-unit ELU hidden layers. The controller runs at 50 Hz.

The full command bounds for forward velocity, lateral velocity and yaw rate are `[-0.5, -0.5, -0.5]` to `[1.2, 0.5, 0.5]`. The command is resampled after 5–10 s, with a standing probability of 0.1. Curriculum groups B and D scale both bounds by 0.25, 0.50, 0.75 and 1.00 at training steps 0, 12,000, 30,000 and 48,000. Dynamics randomization groups C and D sample sliding friction in [0.4, 0.9] and apply multipliers in [0.9, 1.1] to base-link mass, actuator proportional gains and damping. The training interventions include no external pushes.

## Evaluation

Each final checkpoint executes 30 deterministic policy rollouts with evaluation seed 101. The nominal suite contains ten 20 s episodes. The fixed-onset suite contains six signed pushes (±40, ±60 and ±80 N) at 3.0 s and a zero-force control. The timing suite contains ±60 N pushes at six onsets from 2.8 to 3.3 s in 0.1 s increments and a zero-force control. Both push suites command forward velocity of 0.5 m/s and use an 8 s horizon. The 360 total executions include repeated conditions across the two push protocols. Evaluation uses nominal dynamics and disables observation noise.

Each pulse applies a world-frame lateral force at the base link center of mass for 0.20 s. A separate calibration on a free 2 kg body with zero gravity showed that 40 N applied for one 0.01 s physics step changed its velocity by 0.20 m/s. The velocity then held steady. Applying the force at two successive steps produced 0.40 m/s. This verified that the force clears each physics step. The evaluation hook reapplies it before each of the 20 physics substeps, and the recorded step logs were checked for pulse amplitude and duration.

A fall is the upstream body up-axis condition, `up_z <= 0.5`. Recovery requires remaining upright through the 8 s horizon and achieving 25 consecutive post-pulse samples with absolute body-frame lateral velocity at or below 0.20 m/s. Recovery latency is measured from pulse end to the start of that qualifying window. Tracking error is the Euclidean norm of the body-frame planar velocity error. The nominal tracking statistic averages over samples after the first 2 s, then over the five forward commands 0.2, 0.5, 0.8, 1.0 and 1.2 m/s and the three training seeds. Post-fall auto-reset samples are excluded.

## Results

| Group | Forward tracking (m/s), mean ± SD | Fixed-onset falls | Fixed-onset recovery | Timing-sweep falls | Timing-sweep recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 0.141 ± 0.011 | 6/18 | 12/18 | 3/36 | 33/36 |
| B | 0.200 ± 0.028 | 9/18 | 9/18 | 14/36 | 22/36 |
| C | 0.149 ± 0.012 | 8/18 | 10/18 | 7/36 | 29/36 |
| D | 0.209 ± 0.020 | 9/18 | 9/18 | 11/36 | 25/36 |

All 120 nominal rollouts completed without a fall. All 24 fixed-onset trials at ±80 N ended in a fall. Every surviving pushed rollout met the recovery criterion within the recorded horizon. In the selected +60 N, 3.0 s comparison, seed-7 group A regained lateral stability 0.36 s after pulse end, and group D fell at 4.68 s.

The observed timing-sweep fall counts also varied by seed. Across seeds 7, 17 and 27, group A recorded 0, 0 and 3 falls out of 12, while D recorded 3, 5 and 3. The fixed-onset and timing metrics summarize repeated scenarios on the same policies. The independent training replicate count is three per condition.

## Limitations

The comparison covers one robot model, one simulator, flat ground and one training budget. The push suite samples a finite set of onset times and one walking command. Training randomization and the test disturbance target different sources of variability, so these results bound the evidence for lateral-push transfer under this setup. Physical-robot behavior and control effort were outside the measurement protocol.
