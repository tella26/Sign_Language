import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from ast import literal_eval


def train(log_interval, model, train_loader, optimizer, epoch, d_model):
    # set model as training mode
    losses = []
    scores = []
    train_labels = []
    train_preds = []
    """Device Selection"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N_count = 0  # counting total trained sample in one epoch
    for batch_idx, data in enumerate(train_loader):
        X, y, video_ids = data
        # X, y = data[0], data[1]
        # print(X)
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
        X, y = X.to(device), y.to(device).view(-1, )

        N_count += X.size(0)

        optimizer.zero_grad()
        # y = torch.transpose(y.unsqueeze(0),0,1)
        y = y.unsqueeze(0)
        X = torch.transpose(X.squeeze(0),0,1)
        #X = X[:, :, -1]
        y = y.to(torch.float32)
        out = model(X, y)  # output has dim = (batch, number of classes)
        # out = torch.transpose(out,0,1)
        # out = out.max(1, keepdim=True)[1]
        # out = torch.transpose(out,1,0)
        out = torch.round(((out-torch.min(out))/(torch.max(out)-torch.min(out))) * 100)
    
        loss = (compute_loss(out, y)) / 10000

        # loss = F.cross_entropy(output, y)
        losses.append(loss.item())
        out = (out.to(torch.int32)) 
        y = y.to(torch.int32)
        # to compute accuracy
        #y_pred = torch.max(out, 1)  # y_pred != output

        step_score = accuracy_score(y.cpu().data.squeeze().numpy(), out.cpu().data.squeeze().numpy())

        # collect prediction labels
        train_labels.extend(y.cpu().data.squeeze().tolist())
        train_preds.extend(out.cpu().data.squeeze().tolist())

        scores.append(step_score)  # computed on CPU

        loss.backward()

        optimizer.step()

        # show information
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.6f}%'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(),
                100 * step_score))

    return losses, scores, train_labels, train_preds


def validation(model, test_loader, epoch, save_to):
    # set model as testing mode
    model.eval()

    val_loss = []
    all_y = []
    all_y_pred = []
    all_video_ids = []
    all_pool_out = []

    num_copies = 4

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            # distribute data to device
            X, y, video_ids = data
            #X, y = data[0], data[2]
             # X, y = data[0], data[1]
            # print(X)
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
            output = torch.round(((output-torch.min(output))/(torch.max(output)-torch.min(output))) * 100)
            # loss = F.cross_entropy(pool_out, y, reduction='sum')
            loss = (compute_loss(output, y)) / 10000

            val_loss.append(loss.item())  # sum up batch loss
            
                
            # y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(output)
            all_pool_out.extend(output)

    # this computes the average loss on the BATCH
    val_loss = sum(val_loss) / len(val_loss)

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
    print('\nVal. set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), val_loss,
                                                                                        100 * top1acc))

    if save_to:
        # save Pytorch models of best record
        torch.save(model.state_dict(),
                   os.path.join(save_to, 'gcn_epoch{}.pth'.format(epoch + 1)))  # save spatial_encoder
        print("Epoch {} model saved!".format(epoch + 1))

    return val_loss, [top1acc, top3acc, top5acc, top10acc, top30acc], all_y.tolist(), all_y_pred.tolist(), incorrect_video_ids


def compute_loss(out, gt):
    # gt = torch.tensor(gt,dtype= torch.long)
    ce_loss = F.cross_entropy(out, gt)

    return ce_loss


def compute_top_n_accuracy(truths, preds, n):
    best_n = np.argsort(preds, axis=0)[-n:]
    ts = truths
    successes = 0
    for i in range(ts.shape[0]):
        if ts[i] in best_n:
            successes += 1
    return float(successes) / ts.shape[0]
