import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("paper", rc={'font.sans-serif': 'Helvetica',
                             'font.size': 12})

df_green = pd.read_csv('~/Resources/Experiments/dcfnex-12/dcstfn-green/train/history.csv')
df_red = pd.read_csv('~/Resources/Experiments/dcfnex-12/dcstfn-red/train/history.csv')
df_nir = pd.read_csv('~/Resources/Experiments/dcfnex-12/dcstfn-nir/train/history.csv')

df_green = df_green.head(50)
df_red = df_red.head(50)
df_nir = df_nir.head(50)

epoch = df_green['epoch']
metrics = ('loss', 'val_loss')

labels = ('Green', 'Red', 'NIR')
colors = ('green', 'red', 'orange')
linestyles = ('-', '--')

fig, ax = plt.subplots()
for metric, linestyle in zip(metrics, linestyles):
    score = (df_green[metric], df_red[metric], df_nir[metric])
    for i in range(3):
        ax.plot(epoch + 1, score[i], label=labels[i], color=colors[i],
                linestyle=linestyle)

ax.set_yscale('log')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('MSE', fontsize=12)

ax.tick_params(axis='both', which='major', labelsize=9)
ax.tick_params(axis='both', which='minor', labelsize=8)
ax.set_xticks(range(0, epoch.size + 1, 10))
ytick_labels = ax.yaxis.get_ticklabels(minor=True)
ytick_labels[16] = r'$2\times10^4$'
ytick_labels[17] = r'$3\times10^4$'
ytick_labels[18] = r'$4\times10^4$'
ytick_labels[24] = r'$2\times 10^5$'
ax.yaxis.set_ticklabels(ytick_labels, minor=True)
grid_color = (0.95, 0.95, 0.95)
ax.grid(True, color=grid_color)
for n in (20000, 30000, 40000, 200000):
    ax.axhline(y=n, color=grid_color, linewidth=0.6)

for i in range(2):
    ax.plot([], [], color='black', linestyle=linestyles[i])
ax.grid(True)
lines = ax.get_lines()
color_legend = ax.legend(handles=[lines[i] for i in range(3)], labels=labels,
                         loc=1, bbox_to_anchor=(0.967, 1), fontsize=10, frameon=False)
line_legend = ax.legend(handles=[lines[i] for i in range(-2, 0)], labels=('Training', 'Validation'),
                        loc=1, bbox_to_anchor=(0.778, 1), fontsize=10, frameon=False)
ax.add_artist(color_legend)
ax.add_artist(line_legend)
ax.set_title('Loss Curve', fontsize=14, fontweight='bold')

plt.savefig('loss.png', dpi=900)
plt.close()
