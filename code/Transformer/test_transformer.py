import os

from configs import Config
from sign_dataset import Sign_Dataset
import numpy as np
import torch
from sklearn.metrics import accuracy_score


def test(model, test_loader):
    # set model as testing mode
    model.eval()
    
    all_y = []
    all_y_pred = []
    all_video_ids = []
    all_pool_out = []

    num_copies = 4

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            # distribute data to device
            X, y, video_ids = data
            # dictionary that maps integer to its string value 
            label_dict = {}
            # list to store integer labels 
            int_labels = []

            for i in range(len(y)):
                label_dict[i] = y[i]
                int_labels.append(int(label_dict[i]))
                
            #int_labels = literal_eval(int_labels)
            
            y = torch.tensor(int_labels)
            # distribute data to device
            """Device Selection"""
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            X, y = X.to(device), y.to(device).view(-1, )
            
            y = y.unsqueeze(0)
            X = torch.transpose(X.squeeze(0),0,1)          
            y = y.to(torch.float32)

            output = model(X, y)  # output has dim = (batch, number of classes)
            
            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(output)
            all_pool_out.extend(output)

    # compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0).squeeze()
    all_pool_out = torch.stack(all_pool_out, dim=0).cpu().data.numpy()

    # log down incorrectly labelled instances
    incorrect_indices = torch.nonzero(all_y - all_y_pred).squeeze().data
    incorrect_video_ids = [(vid, int(all_y_pred[i].data)) for i, vid in enumerate(all_video_ids) if
                           i in incorrect_indices]

    all_y = all_y.cpu().data.numpy()
    all_y_pred = all_y_pred.cpu().data.numpy()

    all_y_pred = all_y_pred * 100
    all_pool_out = all_pool_out * 100
    all_y_pred = torch.tensor(all_y_pred)
    all_y = torch.tensor(all_y)
    all_pool_out = torch.tensor(all_pool_out)
    
    all_y_pred = all_y_pred.to(torch.int32)
    all_y = all_y.to(torch.int32)
    all_pool_out = all_pool_out.to(torch.int32)
    
    all_y = torch.flatten(all_y)
    all_y_pred = torch.flatten(all_y_pred)
    all_pool_out = torch.flatten(all_pool_out)
    
    # top-k accuracy
    top1acc = accuracy_score(all_y, all_y_pred)
    top3acc = compute_top_n_accuracy(all_y, all_pool_out, 3)
    top5acc = compute_top_n_accuracy(all_y, all_pool_out, 5)
    top10acc = compute_top_n_accuracy(all_y, all_pool_out, 10)
    top30acc = compute_top_n_accuracy(all_y, all_pool_out, 30)

    # show information
    print('Test. set ({:d} samples): top-1 Accuracy: {:.2f}%\n'.format(len(all_y), 100 * top1acc))
    print('Test. set ({:d} samples): top-3 Accuracy: {:.2f}%\n'.format(len(all_y), 100 * top3acc))
    print('Test. set ({:d} samples): top-5 Accuracy: {:.2f}%\n'.format(len(all_y), 100 * top5acc))
    print('Test. set ({:d} samples): top-10 Accuracy: {:.2f}%\n'.format(len(all_y), 100 * top10acc))


def compute_top_n_accuracy(truths, preds, n):
    best_n = np.argsort(preds, axis=0)[-n:]
    ts = truths
    successes = 0
    for i in range(ts.shape[0]):
        if ts[i] in best_n:
            successes += 1
    return float(successes) / ts.shape[0]
