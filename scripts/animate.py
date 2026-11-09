"""Animate recorded control telemetry. Requires matplotlib and FFmpeg."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

ROOT = Path(__file__).resolve().parents[1]
fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True, layout='constrained')
fig.suptitle('Same push, different outcomes | Go2, training seed 7', fontweight='bold')
curves = []
for group, color, label in [('A', '#06766e', 'A  Fixed training'),
                            ('D', '#bd513b', 'D  Curriculum + randomization')]:
    with (ROOT / 'data' / f'example_{group}.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    times = [float(row['time_s']) for row in rows]
    for ax, field in zip(axes, ('vy', 'up_z')):
        values = [float(row[field]) for row in rows]
        line, = ax.plot([], [], lw=1.6, color=color, label=label)
        curves.append((line, times, values))
axes[0].set(ylabel='Lateral velocity (m/s)', ylim=(-1.3, 1.3))
axes[0].axhspan(-.2, .2, color='#06766e', alpha=.08)
axes[0].legend(loc='upper right', frameon=False)
axes[1].set(ylabel='Body up-axis z', xlabel='Simulation time (s)', ylim=(0, 1.08))
axes[1].axhline(.5, color='#bd513b', ls='--', lw=1)
for ax in axes:
    ax.set_xlim(0, 8)
    ax.axvspan(3, 3.2, color='#e5a324', alpha=.30)
    ax.grid(alpha=.16)
    ax.spines[['top', 'right']].set_visible(False)
axes[1].text(3.25, .08, '+60 N for 0.20 s', fontsize=9)


def update(frame):
    elapsed = frame / 25
    for line, times, values in curves:
        end = sum(t <= elapsed for t in times)
        line.set_data(times[:end], values[:end])


if __name__ == '__main__':
    update(200)
    fig.savefig(ROOT / 'media' / 'push_comparison.png', dpi=110)
    animation = FuncAnimation(fig, update, frames=201)
    animation.save(ROOT / 'media' / 'push_comparison.mp4', writer='ffmpeg',
                   fps=25, dpi=90, extra_args=['-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
