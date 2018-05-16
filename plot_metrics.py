import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import argparse
import numpy as np

def plot_fig(data, fig_name, x_label, y_label, title):
    x = np.arange(1, len(data)+1)
    y = np.asarray(data)

    plt.figure()
    plt.plot(x, y, '--bo')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(fig_name)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_file', type=str, required=True)
    args = parser.parse_args()

    ckpt_model = torch.load(args.ckpt_file)

    train_acc = torch.FloatTensor(ckpt_model['tracker']['train_acc'])
    train_acc = train_acc.mean(dim=1).numpy()

    val_acc = torch.FloatTensor(ckpt_model['tracker']['val_acc'])
    val_acc = val_acc.mean(dim=1).numpy()

    train_loss = torch.FloatTensor(ckpt_model['tracker']['train_loss'])
    train_loss = train_loss.mean(dim=1).numpy()

    val_loss = torch.FloatTensor(ckpt_model['tracker']['val_loss'])
    val_loss = val_loss.mean(dim=1).numpy()

    plot_fig(train_acc, 'train_acc.png', 'No. of epochs', 'Train Accuracy', 'Plot of training accuracy vs. no. of epochs')
    plot_fig(val_acc, 'val_acc.png', 'No. of epochs', 'Validation Accuracy', 'Plot of validation accuracy vs. no. of epochs')
    plot_fig(train_loss, 'train_loss.png', 'No. of epochs', 'Train Loss', 'Plot of training loss vs. no. of epochs')
    plot_fig(val_loss, 'val_loss.png', 'No. of epochs', 'Validation Loss', 'Plot of validation loss vs. no. of epochs')



