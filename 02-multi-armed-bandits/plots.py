import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
from scipy.stats import norm

def generate_normal(num=100, e=0.01):
    return norm.ppf(np.linspace(e, 1 - e, num=num, dtype=np.float))

normal = generate_normal()

def plot_bandits(means):
    n = len(means)
    data = np.repeat(normal[:, np.newaxis], n, axis=1) + means
    sns.set(style='whitegrid')
    ax = sns.violinplot(data=data, inner=None)
    ax = sns.scatterplot(data=means, ax=ax)
    ax.set_xlabel('Actions')
    ax.set_ylabel('Q(a)')
    ax.set_xticklabels(np.arange(n) + 1)
    ax.yaxis.set_major_locator(ticker.MultipleLocator())
    plt.show()

def plot_rewards(metric, n, xlim, legend, filename='rewards', title='Rewards'):
    fig, ax1 = plt.subplots()
    fig.canvas.set_window_title(title)
    ax1.set_title(title)
    ax1.plot(metric)
    ax1.legend(legend)
    ax1.set_xlim(0, xlim)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Total Reward')
    ax2 = ax1.twinx()
    ax2.plot(metric / n)
    ax2.set_ylabel('Average Reward')
    fig.tight_layout()
    fig.savefig('results/{}.png'.format(filename))
    plt.show()

def plot_optimal(metric, xlim, legend, filename='optimal_actions', title='Optimal Actions'):
    fig, ax1 = plt.subplots()
    fig.canvas.set_window_title(title)
    ax1.set_title(title)
    ax1.plot(metric * 100.0)
    ax1.legend(legend)
    ax1.set_xlim(0, xlim)
    ax1.set_xlabel('Step')
    ax1.set_ylim(0, 100)
    ax1.set_ylabel('Optimal Action %')
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax2 = ax1.twinx()
    ax2.plot(metric)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Optimal Action P')
    fig.tight_layout()
    fig.savefig('results/{}.png'.format(filename))
    plt.show()

def multicolor_xlabel(ax, list_of_strings, anchorpad=0, **kw):
    # code from: https://stackoverflow.com/a/33162465
    from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker
    boxes = [TextArea(text, textprops=dict(color='C{}'.format(i), ha='left', va='bottom', **kw))
                for i, text in enumerate(list_of_strings)]
    xbox = HPacker(children=boxes, align='center', pad=0, sep=5)
    anchored_xbox = AnchoredOffsetbox(loc='lower center', child=xbox, pad=anchorpad, frameon=False,
                                        bbox_transform=ax.transAxes, borderpad=0.0)
    ax.add_artist(anchored_xbox)

def plot_summary(results, steps):
    fig, ax = plt.subplots()
    fig.set_size_inches((16, 9))
    title = 'Summary'
    fig.canvas.set_window_title(title)
    ax.set_title(title)
    params = []

    for x, y, label, param in results:
        ax.plot(x, y, label=label)
        params.append(param)

    powers = range(-7, 3)
    ticks = [2 ** i for i in powers]
    ax.set_xscale('log', basex=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([r'$\frac{1}{%s}$' % (2 ** -i) if i < 0 else str(2 ** i) for i in powers])
    ax.set_xlim(min(ticks), max(ticks))
    ax.set_xlabel('Parameter Value')
    multicolor_xlabel(ax, params, size=22)
    ax.set_ylabel('Average reward over first {} steps'.format(steps))
    plt.legend()
    fig.savefig('results/summary.png', dpi=100)
    plt.show()