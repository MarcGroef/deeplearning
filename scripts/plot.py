import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

sns.set()
sns.set_context('talk')
palette = sns.color_palette()

def finish(title, xlabel='epochs', ylabel='accuracy', xlim=None, ylim=None, **kwargs):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if 'xticks' in kwargs:
        ticks, labels = kwargs['xticks']
        plt.xticks(ticks, labels)

    plt.subplots_adjust(top=.94, bottom=.13)
    plt.legend()
    plt.show()

def plot_training(name='control', title='Training'):
    n_experiments = 5

    # Load data for all folds
    train_acc = [np.load('../data/' + name + "_train_acc_" + str(expIdx) + '.npy') for expIdx in range(n_experiments)]
    val_acc   = [np.load('../data/' + name + "_val_acc_"   + str(expIdx) + '.npy') for expIdx in range(n_experiments)]
    n_epochs = 25#len(train_acc[0])

    # Find train/val averages and plot
    avg_train_acc = np.average(np.stack(train_acc), axis=0)
    avg_val_acc = np.average(np.stack(val_acc), axis=0)
    plt.plot(avg_train_acc, label='Training')
    plt.plot(avg_val_acc, label='Validation')

    # control()

    # Plot actual accuracies
    plt.scatter(np.stack([list(range(n_epochs)) for _ in range(n_experiments)]), train_acc, s=4, alpha=.5)
    plt.scatter(np.stack([list(range(n_epochs)) for _ in range(n_experiments)]), val_acc  , s=4, alpha=.5)

    finish(title=title, xlim=(0, n_epochs), ylim=(0.775, 1))

def control():
    n_experiments = 5
    train_acc = [np.load('../data/' + 'control' + "_train_acc_" + str(expIdx) + '.npy')[-1] for expIdx in range(n_experiments)]
    val_acc   = [np.load('../data/' + 'control' + "_val_acc_"   + str(expIdx) + '.npy')[-1] for expIdx in range(n_experiments)]
    # print(val_acc)

    control_train, control_val = np.average(train_acc), np.average(val_acc)

    # plot
    plt.axhline(y=control_train, color=palette[0], linestyle=':', alpha=.9, label='Control (train)')
    plt.axhline(y=control_val, color=palette[1], linestyle=':', alpha=.9, label='Control (validation)')

    return control_train, control_val

def plot_dropout(param_values, name='dropout', title='Accuracy after 25 epochs'):
    n_experiments = 5

    train_acc, val_acc = [], []
    for param in param_values:
        # Get final accuracies only for each fold
        train_acc.append([np.load('../data/' + name + '_fc_dropout_rate' + '=' + str(param) + "_train_acc_" + str(expIdx) + '.npy')[-1] for expIdx in range(n_experiments)])
        val_acc.append([np.load('../data/' + name + '_fc_dropout_rate' + '=' + str(param) + "_val_acc_"   + str(expIdx) + '.npy')[-1] for expIdx in range(n_experiments)])

    # Find train/val averages and plot
    avg_train_acc = np.average(np.stack(train_acc), axis=1)
    avg_val_acc = np.average(np.stack(val_acc), axis=1)
    plt.plot(param_values, avg_train_acc, label='Training', color=palette[0])
    plt.plot(param_values, avg_val_acc, label='Validation', color=palette[1])

    # Plot control line
    control()

    # Plot actual accuracies
    train_acc = np.transpose(np.stack(train_acc))
    val_acc = np.transpose(np.stack(val_acc))
    for scalar_run in range(len(val_acc)):
        plt.scatter(param_values, train_acc[scalar_run], color=palette[0], s=4, alpha=.5)
        plt.scatter(param_values, val_acc[scalar_run], color=palette[1], s=4, alpha=.5)

    # param_labels = np.array([float(p) for p in param_values])*1000
    finish(title=title, xlabel='dropout rate', ylim=(0.775, 1))# xticks=[param_values, param_labels]

def plot_l2(param_values, name='l2', title='Accuracy after 25 epochs'):
    n_experiments = 5

    train_acc, val_acc = [], []
    for param in param_values:
        # Get final accuracies only for each fold
        train_acc.append([np.load('../data/' + name + '_l2_scalar' + '=' + str(param) + "_train_acc_" + str(expIdx) + '.npy')[-1] for expIdx in range(n_experiments)])
        val_acc.append([np.load('../data/' + name + '_l2_scalar' + '=' + str(param) + "_val_acc_"   + str(expIdx) + '.npy')[-1] for expIdx in range(n_experiments)])

    # Find train/val averages and plot
    avg_train_acc = np.average(np.stack(train_acc), axis=1)
    avg_val_acc = np.average(np.stack(val_acc), axis=1)
    plt.plot(param_values, avg_train_acc, label='Training', color=palette[0])
    plt.plot(param_values, avg_val_acc, label='Validation', color=palette[1])

    # Plot control line
    control()

    # Plot actual accuracies
    train_acc = np.transpose(np.stack(train_acc))
    val_acc = np.transpose(np.stack(val_acc))
    for scalar_run in range(len(val_acc)):
        plt.scatter(param_values, train_acc[scalar_run], color=palette[0], s=4, alpha=.5)
        plt.scatter(param_values, val_acc[scalar_run], color=palette[1], s=4, alpha=.5)

    param_labels = np.array([float(p) for p in param_values])*1000
    finish(title=title, xlabel='l2 penalty weight (E10-3)', xticks=[param_values, param_labels], ylim=(0.775, 1))# xticks=[param_values, param_labels]

def plot_l1(param_values, name='l1', title='Accuracy after 25 epochs'):
    n_experiments = 5

    train_acc, val_acc = [], []
    for param in param_values:
        # Get final accuracies only for each fold
        train_acc.append([np.load('../data/' + name + '_l1_scalar' + '=' + str(param) + "_train_acc_" + str(expIdx) + '.npy')[-1] for expIdx in range(n_experiments)])
        val_acc.append([np.load('../data/' + name + '_l1_scalar' + '=' + str(param) + "_val_acc_"   + str(expIdx) + '.npy')[-1] for expIdx in range(n_experiments)])

    # Find train/val averages and plot
    avg_train_acc = np.average(np.stack(train_acc), axis=1)
    avg_val_acc = np.average(np.stack(val_acc), axis=1)
    plt.plot(param_values, avg_train_acc, label='Training', color=palette[0])
    plt.plot(param_values, avg_val_acc, label='Validation', color=palette[1])

    # Plot control line
    control()

    # Plot actual accuracies
    train_acc = np.transpose(np.stack(train_acc))
    val_acc = np.transpose(np.stack(val_acc))
    for scalar_run in range(len(val_acc)):
        plt.scatter(param_values, train_acc[scalar_run], color=palette[0], s=4, alpha=.5)
        plt.scatter(param_values, val_acc[scalar_run], color=palette[1], s=4, alpha=.5)

    param_labels = np.array([float(p) for p in param_values])*1000
    finish(title=title, xlabel='l1 penalty weight (E10-3)', xticks=[param_values, param_labels], ylim=(0.775, 1))# xticks=[param_values, param_labels]

if __name__ == "__main__":
    # # plot_training('control')
    # # plot_training('batchnorm')
    # # plot_training('dropout')
    # # params = list(np.arange(0, 0.003, 0.00025))
    # # params = [str(round(param, 6)) for param in params]
    # # plot_parameter(params)

    params = np.array([param*.1 for param in range(10)]).astype(np.float16)
    plot_dropout(params)

    params = list(np.arange(0, 0.003, 0.00025))
    params = [str(round(param, 6)) for param in params]
    plot_l1(params)
    plot_l2(params)

    # plot_training('batchnorm', title='Training with batch normalisation')
    # plot_training('control', title='Training without regularisation')
