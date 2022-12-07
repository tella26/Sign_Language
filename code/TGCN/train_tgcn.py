import logging
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset

import utils
import test_tgcn
from configs import Config
from tgcn_model import GCN_muti_att
from sign_dataset import Sign_Dataset
from train_utils import train, validation
from data import csv_to_json

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def run(split_file, pose_data_root, configs, save_model_to=None):
    epochs = configs.max_epochs
    log_interval = configs.log_interval
    num_samples = configs.num_samples
    hidden_size = configs.hidden_size
    drop_p = configs.drop_p
    num_stages = configs.num_stages

    # setup dataset
    
    train_dataset = Sign_Dataset(index_file_path=split_file, pose_root=pose_data_root, num_samples = num_samples)
    
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                                    shuffle=True)

    val_dataset = Sign_Dataset(index_file_path=split_file, pose_root=pose_val_root, num_samples = num_samples)
    
    val_data_loader = torch.utils.data.DataLoader(dataset=train_dataset , batch_size=configs.batch_size,
                                                  shuffle=True)
    
    test_dataset = Sign_Dataset(index_file_path=split_file, pose_root=pose_test_root,num_samples = num_samples,
                                                test_index_file=split_test_file)
    
    test_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size, shuffle=True)


    logging.info('\n'.join(['Class labels are: '] + [(str(i) + ' - ' + label) for i, label in
                                                     enumerate(train_dataset.label_encoder.classes_)]))
     
    """Device Selection"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # setup the model
    print("Loading the TGCN model.....")
    model = GCN_muti_att(input_feature=2, hidden_feature= 113,
                         num_class=len(train_dataset.label_encoder.classes_), p_dropout=drop_p, num_stage=num_stages).to(device)

    # setup training parameters, learning rate, optimizer, scheduler
    lr = configs.init_lr
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=configs.adam_eps, weight_decay=configs.adam_weight_decay)

    # record training process
    epoch_train_losses = []
    epoch_train_scores = []
    epoch_val_losses = []
    epoch_val_scores = []

    best_test_acc = 0
    # start training
    
    for epoch in range(int(epochs)):
        # train, test model

        print('start training....')
        print('Loading the features to/from features folder....')
        train_losses, train_scores, train_gts, train_preds = train(log_interval, model,
                                                                   train_data_loader, optimizer, epoch)
        print('start validating....')
        val_loss, val_score, val_gts, val_preds, incorrect_samples = validation(model,
                                                                                val_data_loader, epoch,
                                                                                save_to=save_model_to)
        # print('start testing.')
        # val_loss, val_score, val_gts, val_preds, incorrect_samples = validation(model,
        #                                                                         val_data_loader, epoch,
        #                                                                         save_to=save_model_to)

        logging.info('========================\nEpoch: {} Average loss: {:.4f}'.format(epoch, val_loss))
        logging.info('Top-1 acc: {:.4f}'.format(100 * val_score[0]))
        logging.info('Top-3 acc: {:.4f}'.format(100 * val_score[1]))
        logging.info('Top-5 acc: {:.4f}'.format(100 * val_score[2]))
        logging.info('Top-10 acc: {:.4f}'.format(100 * val_score[3]))
        logging.info('Top-30 acc: {:.4f}'.format(100 * val_score[4]))
        logging.debug('mislabelled val. instances: ' + str(incorrect_samples))

        # save results
        epoch_train_losses.append(train_losses)
        epoch_train_scores.append(train_scores)
        epoch_val_losses.append(val_loss)
        epoch_val_scores.append(val_score[0])

        # save all train test results
        np.save('../../output/epoch_training_losses.npy', np.array(epoch_train_losses))
        np.save('../../output/epoch_training_scores.npy', np.array(epoch_train_scores))
        np.save('../../output/epoch_test_loss.npy', np.array(epoch_val_losses))
        np.save('../../output/epoch_test_score.npy', np.array(epoch_val_scores))

        if val_score[0] > best_test_acc:
            best_test_acc = val_score[0]
            best_epoch_num = epoch
            '''
            torch.save(model.state_dict(), os.path.join('../../output/checkpoints', 'gcn_epoch={}_val_acc={}.pth'.format(
                best_epoch_num, best_test_acc)))
            '''
            torch.save(model.state_dict(), os.path.join('../../output/checkpoints', 'ckpt.pth'.format( )))
        else:
            torch.save(model.state_dict(), os.path.join('../../output/checkpoints', 'ckpt.pth'.format( )))

    # For testing
    print('Loading saved model for testing...') 
    checkpoint = 'ckpt.pth'
    checkpoint = torch.load(os.path.join(root, 'output/checkpoints/{}'.format(checkpoint)))
    model.load_state_dict(checkpoint)  
    print('Finished loading model and start testing!')
    test_tgcn.test(model, test_data_loader)
    
    print('Ploting...')
    utils.plot_curves()

    class_names = train_dataset.label_encoder.classes_
    utils.plot_confusion_matrix(train_gts, train_preds, classes=class_names, normalize=True,
                                save_to='output/train-conf-mat')
    utils.plot_confusion_matrix(val_gts, val_preds, classes=class_names, normalize=True, save_to='../../output/val-conf-mat')
    print('Plot saved in the output folder')

if __name__ == "__main__": 
    root = '../../'
    #root = '../../'
    
    subset = 'asl100'
     
    '''
    # csvFilePath_train = r'../../data/WLASL100_train_25fps_normalized.csv'
    csvFilePath_train = r'/content/drive/MyDrive/dataset/Sign-language/WLASL100_train_25fps_normalized.csv' # for google drive
    jsonFilePath = r'../../output/data_json.json'
    csv_to_json(csvFilePath_train, jsonFilePath)
    
    # validation path
    # csvFilePath_val = r'../../data/WLASL100_val_25fps_normalized.csv'
    csvFilePath_val = r'/content/drive/MyDrive/dataset/Sign-language/WLASL100_train_25fps_normalized.csv' # For google drive 
    jsonFilePath_val = r'../../output/data_val_json.json'
    csv_to_json(csvFilePath_val, jsonFilePath_val)
    
    # Testing path
    # csvFilePath_val = r'../../data/WLASL100_test_25fps_normalized.csv'
    # csvFilePath_val = r'/content/drive/MyDrive/dataset/Sign-language/WLASL100_test_25fps_normalized.csv' # For google drive 
    jsonFilePath_val = r'../../output/data_val_json.json'
    csv_to_json(csvFilePath_val, jsonFilePath_val)
    '''
    
    jsonFilePath = r'../../data/data_json.json'
    jsonFilePath_val = r'../../data/data_val_json.json'
    jsonFilePath_test = r'../../data/data_test_json.json'
    
    split_file = os.path.join(root, 'data/{}.json'.format('class_list'))
    split_test_file =  os.path.join(root, 'data/{}.json'.format('class_list'))
    
    pose_data_root = jsonFilePath
    pose_val_root = jsonFilePath_val
    pose_test_root = jsonFilePath_test
    
    config_file = os.path.join(root, 'code/TGCN/configs/{}.ini'.format(subset))
    configs = Config(config_file)

    logging.basicConfig(filename='output/{}.log'.format(os.path.basename(config_file)[:-4]), level=logging.DEBUG, filemode='w+')

    logging.info('Calling main.run()')
    run(split_file=split_file, configs=configs, pose_data_root=pose_data_root)
    logging.info('Finished main.run()')
    utils.plot_curves()
