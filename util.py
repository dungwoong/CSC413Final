import torch
import pandas as pd
import matplotlib.pyplot as plt


def top3_error(predictions, targets):  # even though it says top5, it's top3 ok...
    if len(predictions.shape) == 1:
        predictions = predictions.unsqueeze(1)
    if len(targets.shape) == 1:
        targets = targets.unsqueeze(1)
    with torch.no_grad():
        _, top5_pred = torch.topk(predictions, k=3, dim=1)
        top5_correct = top5_pred.eq(targets.expand(top5_pred.size()))
        correct = top5_correct.float().sum()
        total = targets.size(0)
    return correct.item(), total


def top1_error(predictions, targets):
    if len(predictions.shape) == 1:
        predictions = predictions.unsqueeze(1)
    if len(targets.shape) == 1:
        targets = targets.unsqueeze(1)
    with torch.no_grad():
        _, top1_pred = torch.max(predictions, dim=1)
        top1_correct = top1_pred.unsqueeze(1).eq(targets)
        correct = top1_correct.float().sum()
        total = targets.size(0)
    return correct.item(), total


def plot_training_curve(df, tr_column='loss_train', val_column='loss_test', val_label='Validation Loss',
                        epoch_column='epoch', title='Training vs Validation Loss'):
    """
    You can switch val_label and val_column with top1 accuracy or whatever btw
    """
    # set ggplot style
    # plt.style.use('seaborn')

    # assume you have two arrays 'train_loss' and 'val_loss' containing the loss values

    # create a figure and axis object
    fig, ax = plt.subplots()

    # plot training loss and validation loss
    ax.plot(epoch_column, tr_column, data=df, label='Training Loss')
    ax.plot(epoch_column, val_column, data=df, label=val_label)

    # add title and labels to the plot
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    # add legend to the plot
    ax.legend()
    plt.grid()
    # show the plot
    plt.show()


def plot_loss_acc_curve(*args, **kwargs):
    plot_training_curve(*args, val_column='top1_acc_test', val_label='Top 1 Accuracy(test)', **kwargs)


# top5 should be 100%, top1 should be ~20%
# preds = torch.randn((1024, 5))
# targets = torch.ones((1024, 1))
# c, t = top5_error(preds, targets)
# print(f"Top 5: {c/t}, ({c} / {t})")
# c, t = top1_error(preds, targets)
# print(f"Top 1: {c/t}, ({c} / {t})")
if __name__ == "__main__":
    labels = torch.ones((64, 1))
    preds = torch.zeros((64, 10))
    preds[:, 1] = 1
    print(top1_error(preds, labels))