import librosa
import argparse
import sys
import os
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import yaml
from data_utils_SSL import genSpoof_list,Dataset_ASVspoof2019_train,Dataset_ASVspoof2021_eval
from model import Model
from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed
import pickle as pkl
import re

__author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

def load_split_dict_og(pth):

    d_list = []
    dm_dict = {}
    f_list = []
    p_list = []
    d_meta, file_list, path = pkl.load(open(pth, 'rb'))
    for d,v in d_meta.items():
        # d_list.append(d.replace(" ", ""))
        s = d.replace(" ", "")
        # d_list.append(s)
        # d_meta[s] = d_meta.pop(d)
        dm_dict[s] = v
    
    for f in file_list:
        f_list.append(f.replace(" ", ""))
        # s = f.replace(" ", "")
        # file_list[s] = file_list.pop(f)
    for p in path:
        p_list.append(p.replace(" ", ""))
        # s = p.replace(" ", "")
        # path[s] = path.pop(d)


    return dm_dict, f_list, p_list


def evaluate_accuracy(dev_loader, model, device):
    val_loss = 0.0
    num_total = 0.0
    model.eval()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    for batch_x, batch_y in dev_loader:
        
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        
        batch_loss = criterion(batch_out, batch_y)
        val_loss += (batch_loss.item() * batch_size)
        
    val_loss /= num_total
   
    return val_loss


def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=14, shuffle=False, drop_last=False)
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    
    fname_list = []
    key_list = []
    score_list = []
    
    for batch_x,utt_id in data_loader:
        fname_list = []
        score_list = []  
        batch_size = batch_x.size(0)
        batch_x = batch_x.to(device)
        
        batch_out = model(batch_x)
        
        batch_score = (batch_out[:, 1]  
                       ).data.cpu().numpy().ravel() 
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())
        
        with open(save_path, 'a+') as fh:
            for f, cm in zip(fname_list,score_list):
                fh.write('{} {}\n'.format(f, cm))
        fh.close()   
    print('Scores saved to {}'.format(save_path))

def train_epoch(train_loader, model, lr,optim, device):
    running_loss = 0
    
    num_total = 0.0
    
    model.train()

    #set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    for batch_x, batch_y in train_loader:
       
        batch_size = batch_x.size(0)
        num_total += batch_size
        
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        
        batch_loss = criterion(batch_out, batch_y)
        
        running_loss += (batch_loss.item() * batch_size)
       
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
       
    running_loss /= num_total
    
    return running_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--protocol_path', type=str, default='')
    parser.add_argument('--iteration', type=str, default=None)
    # path where data folder is
    parser.add_argument('--database_path', type=str, default='') 
    # path of where to save the model
    parser.add_argument('--save_model_dir', type=str, default='')
    # give the model a name
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--train_log_path', type=str, default='') 
    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='weighted_CCE')
    # model
    parser.add_argument('--seed', type=int, default=1234, 
                        help='random seed (default: 1234)')
    
    parser.add_argument('--model_path', type=str,
                        default=None, help='Model checkpoint')
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')
    # Auxiliary arguments
    parser.add_argument('--track', type=str, default='DF',choices=['LA', 'PA','DF'], help='LA/PA/DF')
    parser.add_argument('--eval_output', type=str, default=None,
                        help='Path to save the evaluation result')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='eval mode')
    parser.add_argument('--is_eval', action='store_true', default=False,help='eval database')
    parser.add_argument('--eval_part', type=int, default=0)
    # backend options
    parser.add_argument('--cudnn-deterministic-toggle', action='store_false', \
                        default=True, 
                        help='use cudnn-deterministic? (default true)')    
    
    parser.add_argument('--cudnn-benchmark-toggle', action='store_true', \
                        default=False, 
                        help='use cudnn-benchmark? (default false)') 


    ##===================================================Rawboost data augmentation ======================================================================#

    parser.add_argument('--algo', type=int, default=3, 
                    help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .default=0]')

    # LnL_convolutive_noise parameters 
    parser.add_argument('--nBands', type=int, default=5, 
                    help='number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]')
    parser.add_argument('--minF', type=int, default=20, 
                    help='minimum centre frequency [Hz] of notch filter.[default=20] ')
    parser.add_argument('--maxF', type=int, default=8000, 
                    help='maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]')
    parser.add_argument('--minBW', type=int, default=100, 
                    help='minimum width [Hz] of filter.[default=100] ')
    parser.add_argument('--maxBW', type=int, default=1000, 
                    help='maximum width [Hz] of filter.[default=1000] ')
    parser.add_argument('--minCoeff', type=int, default=10, 
                    help='minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]')
    parser.add_argument('--maxCoeff', type=int, default=100, 
                    help='maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]')
    parser.add_argument('--minG', type=int, default=0, 
                    help='minimum gain factor of linear component.[default=0]')
    parser.add_argument('--maxG', type=int, default=0, 
                    help='maximum gain factor of linear component.[default=0]')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5, 
                    help=' minimum gain difference between linear and non-linear components.[default=5]')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20, 
                    help=' maximum gain difference between linear and non-linear components.[default=20]')
    parser.add_argument('--N_f', type=int, default=5, 
                    help='order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]')

    # ISD_additive_noise parameters
    parser.add_argument('--P', type=int, default=10, 
                    help='Maximum number of uniformly distributed samples in [%].[default=10]')
    parser.add_argument('--g_sd', type=int, default=2, 
                    help='gain parameters > 0. [default=2]')

    # SSI_additive_noise parameters
    parser.add_argument('--SNRmin', type=int, default=10, 
                    help='Minimum SNR value for coloured additive noise.[default=10]')
    parser.add_argument('--SNRmax', type=int, default=40, 
                    help='Maximum SNR value for coloured additive noise.[default=40]')
    
    ##===================================================Rawboost data augmentation ======================================================================#
    
    args = parser.parse_args()
    if not os.path.exists(args.save_model_dir):
        os.mkdirs(args.save_model_dir)
    
 
    #make experiment reproducible
    set_random_seed(args.seed, args)
    
    #define model saving path
    model_save_path = os.path.join(args.save_model_dir)

    #set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    
    #GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))
    
    model = Model(args,device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    model = (model).to(device)
    
    print('nb_params:',nb_params)

    #set Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)

    start_epoch = 0
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path,map_location=device))
        print('Model loaded : {}'.format(args.model_path))
        start_epoch = int(os.path.basename(args.model_path).split("_")[-1].replace(".pth", ""))
        
    # define train dataloader
    # d_label_trn,file_train = genSpoof_list( dir_meta = args.protocol_path,is_train=True,is_eval=False)
    
    d_label_trn,file_train,path = load_split_dict_og(args.protocol_path)
    # d_label_trn,file_train,path,name = load_split_dict_extra_training(iterations)
    name = args.model_name
    print('\tno. of training trials',len(file_train))

    model_tag = 'model_{}'.format(name)
    model_save_path = os.path.join(args.save_model_dir, model_tag)

    #set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    train_set=Dataset_ASVspoof2019_train(args,list_IDs = file_train,labels = d_label_trn,base_dir = os.path.join(args.database_path),algo=args.algo)

    train_loader = DataLoader(train_set, batch_size=args.batch_size,num_workers=8, shuffle=True,drop_last = True)

    del train_set,d_label_trn


    # Training
    num_epochs = args.num_epochs
    writer = SummaryWriter(os.path.join(args.train_log_path, args.model_name))
    
    for epoch in range(start_epoch, num_epochs, 1):
        
        running_loss = train_epoch(train_loader,model, args.lr,optimizer, device)
        # val_loss = evaluate_accuracy(dev_loader, model, device)
        # writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('loss', running_loss, epoch)
        print('\n{} - {} '.format(epoch, running_loss))
        torch.save(model.state_dict(), os.path.join(
            model_save_path, 'epoch_{}.pth'.format(epoch)))
