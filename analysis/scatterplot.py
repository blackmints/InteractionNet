import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from model.dataset import Dataset
from model.layers import *
from scipy.stats import gaussian_kde

# Settings
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

plt.rcParams['font.size'] = 6
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['lines.linewidth'] = 1


def scatterplot_and_histogram(model, hyper, target='test'):
    tick = 2.0

    # Make folder
    fig_path = "../analysis/{}".format(model)
    if not os.path.isdir(fig_path):
        os.mkdir(fig_path)
    fig_path = "../analysis/{}/{}".format(model, hyper)
    if not os.path.isdir(fig_path):
        os.mkdir(fig_path)

    # Load results
    base_path = "../result/{}/{}/".format(model, hyper)
    for trial in range(20):
        path = base_path + 'trial_{:02d}/'.format(trial)

        # Load model
        custom_objects = {'NodeEmbedding': NodeEmbedding,
                          'GraphConvolution': GraphConvolution,
                          'Normalize': Normalize,
                          'GlobalPooling': GlobalPooling}
        model = load_model(path + 'best_model.h5', custom_objects=custom_objects)

        # Load data
        data = np.load(path + 'data_split.npz')
        dataset = Dataset('refined', 5)
        dataset.split_by_idx(32, data['train'], data['valid'], data['test'])
        data.close()

        # Predict
        if target == 'train':
            pred_y = model.predict(dataset.train, steps=dataset.train_step, verbose=0).flatten()
            true_y = dataset.train_y
            if len(pred_y) <= len(true_y):
                true_y = true_y[:len(pred_y)]
            else:
                pred_y = pred_y[:len(true_y)]
        elif target == 'valid':
            pred_y = model.predict(dataset.valid, steps=dataset.valid_step, verbose=0).flatten()
            true_y = dataset.valid_y
            if len(pred_y) <= len(true_y):
                true_y = true_y[:len(pred_y)]
            else:
                pred_y = pred_y[:len(true_y)]
        else:
            pred_y = model.predict(dataset.test, steps=dataset.test_step, verbose=0).flatten()
            true_y = dataset.test_y
            if len(pred_y) <= len(true_y):
                true_y = true_y[:len(pred_y)]
            else:
                pred_y = pred_y[:len(true_y)]

        diff_y = true_y - pred_y

        # Draw figure
        fig, axes = plt.subplots(1, 2)
        fig.set_size_inches(5, 2)

        # Scatterplot and trend line
        axes[0].scatter(true_y, pred_y, c='#000000ff', s=2, linewidth=1)
        axes[0].set_aspect('equal', 'box')
        x_min, x_max, y_min, y_max = axes[0].axis()
        axes[0].set_xlim(min(x_min, y_min), max(x_max, y_max))
        axes[0].set_ylim(min(x_min, y_min), max(x_max, y_max))
        axes[0].set_xticks(np.arange(int(min(x_min, y_min)), max(x_max, y_max), tick))
        axes[0].set_yticks(np.arange(int(min(x_min, y_min)), max(x_max, y_max), tick))
        axes[0].spines['right'].set_visible(False)
        axes[0].spines['top'].set_visible(False)

        x_min, x_max, y_min, y_max = axes[0].axis()
        trend_z = np.polyfit(true_y, pred_y, 1)
        trend_p = np.poly1d(trend_z)
        axes[0].plot([x_min, x_max], [trend_p(x_min), trend_p(x_max)], color='black', alpha=1, linestyle="-")

        # Histogram
        bins = np.linspace(np.floor(np.min(diff_y)), np.ceil(np.max(diff_y)), 25)
        n, x, _ = axes[1].hist(diff_y, bins=bins, histtype='bar', density=True, color='orange')
        density = gaussian_kde(diff_y)
        axes[1].plot(x, density(x), linestyle='-', color="black")

        x_min, x_max, y_min, y_max = axes[1].axis()
        x_limit = np.ceil(max(np.absolute(x_min), x_max))
        axes[1].set_xlim(-x_limit, x_limit)
        axes[1].set_ylim(0, 0.5)
        axes[1].set_xticks(np.arange(-x_limit, x_limit + 0.01, tick))
        asp = np.diff(axes[1].get_xlim())[0] / np.diff(axes[1].get_ylim())[0]
        axes[1].set_aspect(asp)
        axes[1].spines['right'].set_visible(False)
        axes[1].spines['top'].set_visible(False)

        # Save analysis
        fig_name = fig_path + '/{}_histogram_{}.png'.format(trial, target)
        plt.savefig(fig_name, dpi=600)
        print('Histogram saved on {}'.format(fig_name))
        plt.clf()


if __name__ == "__main__":
    scatterplot_and_histogram('InteractionNetCNC', '__YOUR_LOG_PATH__', 'test')
