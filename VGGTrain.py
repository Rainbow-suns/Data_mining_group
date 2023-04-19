import datetime

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from Common_Operation import drawLines, initial_train, initial_list, train, valid, drawLR


def VGGTrain(lr=9e-5, bs=6):
    algorithm = 'VGGnet'
    epochs = 25
    learning_rate = lr
    batch_size = bs
    best_val_loss = np.inf
    train_dataloader, valid_dataloader, loss_fn, model = initial_train(algorithm, batch_size)  # algorithm, batch_size, lr

    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.65)
    scheduler = CosineAnnealingLR(optimizer, T_max=8)
    train_lossList, val_accList, val_lossList = initial_list()
    learning_rates = []
    startTime = datetime.datetime.now()
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loss = train(train_dataloader, model, loss_fn, optimizer)
        val_acc, val_loss = valid(valid_dataloader, model, loss_fn)
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            torch.save(model, './TrainedModel/VGGnet.pth')

        #train_lossList.append(train_loss)
        val_lossList.append(val_loss)
        val_accList.append(val_acc)
        scheduler.step()
        learning_rates.append(scheduler.get_last_lr()[0])

    endTime = datetime.datetime.now()
    print("Done!")
    print("VGGnet training time: " + str(endTime - startTime) + ".")
    torch.save(model, './TrainedModel/VGGnetLast.pth')

    #drawLines(epochs, train_lossList, val_lossList, val_accList, algorithm, lr, bs)
    drawLR(epochs, learning_rates)


if __name__ == '__main__':
    VGGTrain()
