import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from deepke.utils import to_one_hot


def train(epoch, device, dataloader, model, optimizer, criterion, config,total_loss=[]):
    model.train()
    # total_loss = []

    for batch_idx, (*x, y) in enumerate(dataloader, 1):
        x = [i.to(device) for i in x]
        y = y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)

        if model.model_name == 'Capsule':
            y = to_one_hot(y, config.relation_type)
            loss = model.loss(y_pred, y)
        else:
            loss = criterion(y_pred, y)

        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())

        # logging
        data_cal = len(dataloader.dataset) if batch_idx == len(dataloader) else batch_idx * len(y)
        if (config.training.train_log
                and batch_idx % config.training.log_interval == 0) or batch_idx == len(dataloader):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, data_cal, len(dataloader.dataset),
                                                                           100. * batch_idx / len(dataloader),
                                                                           loss.item()))

    # # plot
    # if config.training.show_plot:
    #     plt.plot(total_loss)
    #     plt.title('loss')
    #     plt.show()
    return total_loss
    

def plot(total_loss,config):
        # plot
    # if config.training.show_plot:
    plt.plot(total_loss)
    plt.title('loss')
    plt.show()


def validate(dataloader, model, device, config):
    model.eval()

    with torch.no_grad():
        total_y_true = np.empty(0)
        total_y_pred = np.empty(0)
        for batch_idx, (*x, y) in enumerate(dataloader, 1):
            x = [i.to(device) for i in x]
            y = y.to(device)
            y_pred = model(x)

            if model.model_name == 'Capsule':
                y_pred = model.predict(y_pred)
            else:
                y_pred = y_pred.argmax(dim=-1)

            try:
                y, y_pred = y.numpy(), y_pred.numpy()
            except:
                y, y_pred = y.cpu().numpy(), y_pred.cpu().numpy()
            total_y_true = np.append(total_y_true, y)
            total_y_pred = np.append(total_y_pred, y_pred)

        total_f1 = []
        for average in config.training.f1_norm:
            p, r, f1, _ = precision_recall_fscore_support(total_y_true, total_y_pred, average=average)
            print(f' {average} metrics: [p: {p:.4f}, r:{r:.4f}, f1:{f1:.4f}]')
            total_f1.append(f1)

    return total_f1
