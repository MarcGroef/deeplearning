import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

sns.set()
sns.set_context('paper')

def plot_training(name='control'):
    n_experiments = 10
    n_epochs = 30

    # Load data for all folds
    train_acc = [np.load('../data/' + name + "_train_acc_" + str(expIdx) + '.npy') for expIdx in range(n_experiments)]
    val_acc   = [np.load('../data/' + name + "_val_acc_"   + str(expIdx) + '.npy') for expIdx in range(n_experiments)]

    # Find train/val averages and plot
    avg_train_acc = np.average(np.stack(train_acc), axis=0)
    avg_val_acc = np.average(np.stack(val_acc), axis=0)
    plt.plot(avg_train_acc, label='Training')
    plt.plot(avg_val_acc, label='Validation')

    # Plot actual accuracies
    plt.scatter(np.stack([list(range(30)) for _ in range(n_experiments)]), train_acc, s=4, alpha=.5)
    plt.scatter(np.stack([list(range(30)) for _ in range(n_experiments)]), val_acc  , s=4, alpha=.5)

    plt.title(name)
    plt.xlabel('epoch')
    plt.ylabel('proportion correct')
    plt.xlim((0, 30))
    plt.ylim((0, 1))

    # Unique labels
    # handles, labels = plt.gca().get_legend_handles_labels()
    # by_label = OrderedDict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys())
    plt.legend()
    plt.show()

def plot_parameter(param_values, name='l2', title='Accuracies after 30 epochs'):
    n_experiments = 10
    # n_epochs = 30

    train_acc = []
    val_acc = []
    for param in param_values:
        # Get final accuracies only for each fold
        train_acc.append([np.load('../data/' + name + '_l2_scalar' + '=' + str(param) + "_train_acc_" + str(expIdx) + '.npy')[-1] for expIdx in range(n_experiments)])
        val_acc.append([np.load('../data/' + name + '_l2_scalar' + '=' + str(param) + "_val_acc_"   + str(expIdx) + '.npy')[-1] for expIdx in range(n_experiments)])

    print(val_acc)

    # Find train/val averages and plot
    avg_train_acc = np.average(np.stack(train_acc), axis=1)
    avg_val_acc = np.average(np.stack(val_acc), axis=1)
    plt.plot(param_values, avg_train_acc, label='Training')
    plt.plot(param_values, avg_val_acc, label='Validation')

    # Plot actual accuracies
    train_acc = np.transpose(np.stack(train_acc))
    val_acc = np.transpose(np.stack(val_acc))
    plt.scatter(np.stack([param_values for _ in range(n_experiments)]), train_acc, s=4, alpha=.5)
    plt.scatter(np.stack([param_values for _ in range(n_experiments)]), val_acc  , s=4, alpha=.5)

    plt.title(title)
    plt.xlabel('l2 scalar')
    plt.ylabel('proportion correct')
    # plt.xlim((0, 30))
    plt.ylim((0, 1))
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # plot_training('control')
    # plot_training('batchnorm')
    plot_parameter([0.01,0.007,0.004,0.001,0.0007])
