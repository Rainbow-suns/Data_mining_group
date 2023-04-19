import itertools

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

from Dataset import dataset
from common import seed
from model import Alex, resnet, VGGSeries


def initial_train(algorithm, bs):
    TD = dataset.createBinaryDataset("./data/train")
    td = DataLoader(TD, batch_size=bs, shuffle=True, worker_init_fn=seed.worker_init_fn)
    VD = dataset.createBinaryDataset("./data/valid")
    vd = DataLoader(VD, batch_size=bs, shuffle=True, worker_init_fn=seed.worker_init_fn)
    if algorithm == 'AlexNet':
        model = Alex.AlexNet()
    elif algorithm == 'Resnet':
        model = resnet.resnet34()
    elif algorithm == 'VGGnet':
        model = VGGSeries.VGG_11()
    else:
        print('Algorithm not support!')
    lf = nn.CrossEntropyLoss()
    return td, vd, lf, model


def initial_test(algorithm, bs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    model = torch.load('./TrainedModel/' + algorithm + '.pth')
    model.to(device)
    TD = dataset.createBinaryDataset("./data/test")
    test_dataloader = DataLoader(TD, batch_size=bs, shuffle=True, worker_init_fn=seed.worker_init_fn)
    return test_dataloader, model


def initial_list():
    tl = []
    va = []
    vl = []
    return tl, va, vl


def train(td, model, lf, op):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    model = model.to(device)
    model.train()
    size = len(td.dataset)
    train_loss = 0.0
    for batch, (X, y) in enumerate(td):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = lf(pred, y)
        # Backpropagation
        op.zero_grad()
        loss.backward()
        op.step()
        loss = loss.item()
        train_loss += loss
        if batch % 100 == 0:
            current = batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return (train_loss * td.batch_size) / size


def confusion_matrix(predicts, labels, conf_matrix):
    # confusion_matrix method is implemented based on the code from
    # https://blog.csdn.net/qq_18617009/article/details/103345308
    for p, t in zip(predicts, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    # plot_confusion_matrix method is implemented based on the code from
    # https://blog.csdn.net/qq_18617009/article/details/103345308
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    plt.axis("equal")

    ax = plt.gca()
    left, right = plt.xlim()
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        plt.text(i, j, num,
                 verticalalignment='center',
                 horizontalalignment="center",
                 color="white" if num > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def test(dataloader, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    concrete_types = ['Negative', 'Positive']
    conf_matrix = torch.zeros(2, 2)
    size = len(dataloader.dataset)
    model.eval()
    correct = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            predict = torch.max(pred, 1)[1]
            conf_matrix = confusion_matrix(predict, labels=y, conf_matrix=conf_matrix)

    conf_matrix = conf_matrix.numpy()
    tp = conf_matrix[1][1]
    fp = conf_matrix[0][1]
    fn = conf_matrix[1][0]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1score = (2 * precision * recall) / (precision + recall)
    correct /= size
    print(f"Accuracy: {correct:>0.3f}")
    print(f"Precison: {precision:>0.3f}")
    print(f"Recall: {recall:>0.3f}")
    print(f"F1Score: {f1score:>0.3f}")
    plot_confusion_matrix(conf_matrix, classes=concrete_types, normalize=False, title='Normalized confusion matrix')

    return correct


def valid(dl, model, lf):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    size = len(dl.dataset)
    num_batches = len(dl)
    model.eval()
    valid_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dl:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            valid_loss += lf(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    valid_loss /= num_batches
    correct /= size
    print(f"Valid Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {valid_loss:>8f} \n")
    return correct, valid_loss


def drawLines(epochs, tl, vl, va, algorithm, lr, bs):
    x1 = np.arange(0, epochs) + 1
    x2 = np.arange(0, epochs) + 1
    y1 = tl
    y2 = vl
    y3 = va
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, label='Train Loss')
    plt.plot(x1, y2, label='Valid Loss')
    plt.legend()
    plt.xlabel('epoches')
    plt.title('Loss')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y3, label='Valid Accuracy')
    plt.legend()
    plt.xlabel('epoches')
    plt.title('Accuracy')
    plt.tight_layout()
    plt.savefig('./TrainResults/Result for ' + algorithm + "_lr_%.5f_bs_%d.png" % (lr, bs))
    plt.show()


def drawLR(epochs, lr):
    x1 = np.arange(0, epochs) + 1
    y1 = lr
    plt.plot(x1, y1)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.show()
