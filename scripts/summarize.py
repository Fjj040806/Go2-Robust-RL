"""Recompute the headline statistics and optionally render the figures."""
import argparse
import csv
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (7, 17, 27)
SPEEDS = (0.2, 0.5, 0.8, 1.0, 1.2)
COLORS = ('#06766e', '#4374ac', '#9262a6', '#bd513b')


def summarize(rows):
    output = {}
    for group in 'ABCD':
        subset = [r for r in rows if r['group'] == group]
        errors = []
        for seed in SEEDS:
            forward = [r for r in subset if int(r['train_seed']) == seed
                       and r['protocol'] == 'nominal' and r['scenario'].startswith('forward_')
                       and float(r['cmd_vx']) in SPEEDS]
            assert len(forward) == len(SPEEDS)
            errors.append(mean(float(r['mean_planar_error_m_s']) for r in forward))
        counts = {}
        for protocol in ('fixed', 'timing'):
            pushed = [r for r in subset if r['protocol'] == protocol and float(r['force_N']) != 0]
            counts[protocol] = (sum(int(r['fallen']) for r in pushed),
                                sum(bool(r['recovery_latency_s']) for r in pushed), len(pushed))
        output[group] = (errors, counts)
    return output


def plot(results):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), layout='constrained')
    groups = list(results)
    for x, (group, color) in enumerate(zip(groups, COLORS)):
        errors, counts = results[group]
        axes[0].errorbar(x, mean(errors), yerr=stdev(errors), fmt='o', color=color, capsize=4)
        axes[0].scatter([x-.13, x-.07, x+.10], errors, color=color, alpha=.45, s=22)
        for protocol, offset, alpha in [('fixed', -.17, 1), ('timing', .17, .48)]:
            falls, _, n = counts[protocol]
            axes[1].bar(x+offset, 100*falls/n, width=.30, color=color, alpha=alpha,
                        label=protocol.title() if x == 0 else None)
    axes[0].set(title='Forward speed tracking', ylabel='Mean planar error (m/s)', ylim=(0,.27))
    axes[1].set(title='Falls after lateral pushes', ylabel='Fall rate (%)', ylim=(0,60))
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.set_xticks(range(4), groups)
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    fig.savefig(ROOT / 'media' / 'results.png', dpi=130)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    with (ROOT / 'data' / 'episodes.csv').open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 360
    results = summarize(rows)
    for group, (errors, counts) in results.items():
        print(f'{group}  {mean(errors):.6f} ± {stdev(errors):.6f} m/s  {counts}')
    if args.plot:
        plot(results)
