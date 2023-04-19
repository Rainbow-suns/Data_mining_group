import datetime

import numpy as np
import torch

from Common_Operation import drawLines, initial_train, initial_list, train, valid
from torch.optim.lr_scheduler import CosineAnnealingLR


def ResnetTrain(lr=9e-5, bs=6):
    algorithm = 'Resnet'
    epochs = 25
    learning_rate = lr
    batch_size = bs
    best_val_loss = np.inf
    train_dataloader, valid_dataloader, loss_fn, model = initial_train(algorithm, batch_size)  # algorithm, batch_size, lr

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=16)
    train_lossList, val_accList, val_lossList = initial_list()
    startTime = datetime.datetime.now()
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loss = train(train_dataloader, model, loss_fn, optimizer)
        val_acc, val_loss = valid(valid_dataloader, model, loss_fn)
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            torch.save(model, './TrainedModel/Resnet.pth')

        train_lossList.append(train_loss)
        val_lossList.append(val_loss)
        val_accList.append(val_acc)
        scheduler.step()
    endTime = datetime.datetime.now()
    print("Done!")
    print("ResNet18 training time: " + str(endTime - startTime) + ".")
    torch.save(model, './TrainedModel/LastResnet.pth')

    drawLines(epochs, train_lossList, val_lossList, val_accList, algorithm, lr, bs)


if __name__ == '__main__':
    ResnetTrain()
