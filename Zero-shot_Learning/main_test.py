from data_provider.data_factory import data_provider
from utils.tools_tsf import EarlyStopping, adjust_learning_rate, visual, vali, test
from tqdm import tqdm
from models.GPT4TS import GPT4TS
# from models.PatchTST import PatchTST
# from models.DLinear import DLinear
# from models.TimesNet import TimesNet
# from models.ETSformer import ETSformer
# from models.Stationary import Stationary
# from models.LightTS import LightTS
# from models.FEDformer import FEDformer
# from models.Autoformer import Autoformer
# from models.Informer import Informer
# from models.Reformer import Reformer

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

import argparse
import random

warnings.filterwarnings('ignore')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='GPT4TS')

parser.add_argument('--task_name', type=str, default='short_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--model_id', type=str, required=True, default='test')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

parser.add_argument('--root_path', type=str, default='./dataset/traffic/')
parser.add_argument('--data_path', type=str, default='traffic.csv')
parser.add_argument('--data', type=str, default='custom')
parser.add_argument('--features', type=str, default='M')
parser.add_argument('--freq', type=int, default=1)
parser.add_argument('--target', type=str, default='OT')
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--percent', type=int, default=10)

parser.add_argument('--seq_len', type=int, default=512)
parser.add_argument('--pred_len', type=int, default=96)
parser.add_argument('--label_len', type=int, default=48)

parser.add_argument('--test_root_path', type=str, default='./dataset/traffic/')
parser.add_argument('--test_data_path', type=str, default='traffic.csv')
parser.add_argument('--test_seq_len', type=int, default=512)
parser.add_argument('--test_pred_len', type=int, default=96)
parser.add_argument('--test_label_len', type=int, default=48)
parser.add_argument('--test_batch_size', type=int, default=512)

parser.add_argument('--decay_fac', type=float, default=0.75)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--train_epochs', type=int, default=10)
parser.add_argument('--lradj', type=str, default='type1')
parser.add_argument('--kernel_size', type=int, default=25)

parser.add_argument('--gpt_layers', type=int, default=3)
parser.add_argument('--is_gpt', type=int, default=1)
parser.add_argument('--e_layers', type=int, default=3)
parser.add_argument('--d_layers', type=int, default=3)
parser.add_argument('--d_model', type=int, default=768)
parser.add_argument('--n_heads', type=int, default=16)
parser.add_argument('--d_ff', type=int, default=512)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--enc_in', type=int, default=862)
parser.add_argument('--dec_in', type=int, default=862)
parser.add_argument('--c_out', type=int, default=862)
parser.add_argument('--patch_size', type=int, default=16)
parser.add_argument('--stride', type=int, default=8)

parser.add_argument('--loss_func', type=str, default='mse')
parser.add_argument('--pretrain', type=int, default=1)
parser.add_argument('--freeze', type=int, default=1)
parser.add_argument('--model', type=str, default='model')

parser.add_argument('--itr', type=int, default=1)
parser.add_argument('--print_int', type=int, default=10)
parser.add_argument('--max_len', type=int, default=-1)
parser.add_argument('--tmax', type=int, default=10)
parser.add_argument('--cos', type=int, default=0)
parser.add_argument('--notrain', type=int, default=0)
parser.add_argument('--train_all', type=int, default=0)

parser.add_argument('--look_back_num', type=int, default=12)
parser.add_argument('--proj_hid', type=int, default=8)
parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
parser.add_argument('--season', type=str, default='hourly')
parser.add_argument('--activation', type=str, default='gelu', help='activation')

parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                    help='hidden layer dimensions of projector (List)')
parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--modes', type=int, default=32, help='modes to be selected random 64')

args = parser.parse_args()

SEASONALITY_MAP = {
   "minutely": 1440,
   "10_minutes": 144,
   "half_hourly": 48,
   "hourly": 24,
   "daily": 7,
   "weekly": 1,
   "monthly": 12,
   "quarterly": 4,
   "yearly": 1
}

mses = []
maes = []
smapes = []
mapes = []
nds = []

batch_size = args.batch_size
root_path = args.root_path
data_path = args.data_path
seq_len = args.seq_len
pred_len = args.pred_len

if args.notrain:
    args.train_epochs = 0

for ii in range(args.itr):

    setting = '{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_gl{}_df{}_eb{}_itr{}'.format(args.model_id, args.seq_len, args.label_len, args.pred_len,
                                                                    args.d_model, args.n_heads, args.e_layers, args.gpt_layers, 
                                                                    args.d_ff, args.embed, ii)
    path = os.path.join(args.checkpoints, setting)
    if not os.path.exists(path):
        os.makedirs(path)

    if args.freq == 0:
        args.freq = 'h'

    args.root_path = root_path
    args.data_path = data_path
    args.seq_len = seq_len
    args.pred_len = pred_len
    args.batch_size = batch_size
    
    train_data, train_loader = data_provider(args, 'train', train_all=args.train_all)

    args.batch_size = args.test_batch_size
    vali_data, vali_loader = data_provider(args, 'val', drop_last_test=False, train_all=args.train_all)

    args.root_path = args.test_root_path
    args.data_path = args.test_data_path
    args.seq_len = args.test_seq_len
    args.pred_len = args.test_pred_len
    test_data, test_loader = data_provider(args, 'test', drop_last_test=False)

    if args.freq != 'h':
        args.freq = SEASONALITY_MAP[test_data.freq]
        print("freq = {}".format(args.freq))
    args.freq = args.season

    device = torch.device('cuda:0')

    time_now = time.time()
    train_steps = len(train_loader)


    if args.model == 'PatchTST':
        model = PatchTST(args, device)
        model.to(device)
    elif args.model == 'DLinear':
        model = DLinear(args, device)
        model.to(device)
    elif args.model == 'TimesNet':
        model = TimesNet(args, device)
        model.to(device)
    elif args.model == 'ETSformer':
        model = ETSformer(args, device)
        model.to(device)
    elif args.model == 'LightTS':
        model = LightTS(args, device)
        model.to(device)
    elif args.model == 'Stationary':
        model = Stationary(args, device)
        model.to(device)
    elif args.model == 'FEDformer':
        model = FEDformer(args, device)
        model.to(device)
    elif args.model == 'Autoformer':
        model = Autoformer(args, device)
        model.to(device)
    elif args.model == 'Informer':
        model = Informer(args, device)
        model.to(device)
    elif args.model == 'Reformer':
        model = Reformer(args, device)
        model.to(device)
    else:
        model = GPT4TS(args, device)
        model.to(device)
    params = model.parameters()
    model_optim = torch.optim.Adam(params, lr=args.learning_rate)
    
    early_stopping = EarlyStopping(patience=400, verbose=True)
    if args.loss_func == 'mse':
        criterion = nn.MSELoss()
    elif args.loss_func == 'smape':
        class SMAPE(nn.Module):
            def __init__(self):
                super(SMAPE, self).__init__()
            def forward(self, pred, true):
                return torch.mean(200 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true) + 1e-8))
        criterion = SMAPE()
    elif args.loss_func == 'mape':
        class MAPE(nn.Module):
            def __init__(self):
                super(MAPE, self).__init__()
            def forward(self, pred, true):
                return torch.mean(100 * torch.abs(pred - true) / (torch.abs(true) + 1e-8))
        criterion = MAPE()
    elif args.loss_func == 'nd':
        class ND(nn.Module):
            def __init__(self):
                super(ND, self).__init__()
            def forward(self, pred, true):
                return torch.mean(torch.abs(pred - true)) / (torch.mean(torch.abs(true)) + 1e-8)
        criterion = ND()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=args.tmax, eta_min=1e-8)

    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []

        epoch_time = time.time()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
            
            iter_count += 1
            model_optim.zero_grad()
            batch_x = batch_x.float().to(device)

            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            
            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
            
            if args.model == 'Stationary' or args.model == 'Autoformer' or args.model == 'Informer':
                outputs = model(batch_x, batch_x_mark, dec_inp, None)
            elif args.model == 'Reformer':
                outputs = model(batch_x, None, dec_inp, None)
            else:
                outputs = model(batch_x, batch_x_mark)

            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :].to(device)
            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())

            loss.backward()
            model_optim.step()

            if (i + 1) % args.print_int == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))

                # test_loss = vali(model, test_data, test_loader, criterion, args, device)
                # print("Test Loss: {0:.7f}".format(test_loss))
                # mse, mae, smape, mape, nd = test(model, test_data, test_loader, args, device, ii)
                # print("smape = {}, mape = {}, nd = {}".format(smape, mape, nd))
                
                iter_count = 0
                time_now = time.time()
                
            # if (i + 1) == 100:
            #     break
        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

        train_loss = np.average(train_loss)
        vali_loss = vali(model, vali_data, vali_loader, criterion, args, device, ii)
        # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
        #     epoch + 1, train_steps, train_loss, vali_loss))
        test_loss = vali(model, test_data, test_loader, criterion, args, device, ii)
        # mse, mae, smape, mape, nd = test(model, test_data, test_loader, args, device, ii)
        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
            epoch + 1, train_steps, train_loss, vali_loss, test_loss))
        # print("smape = {}, mape = {}".format(smape, mape))

        early_stopping(vali_loss, model, path)
        if args.cos:
            scheduler.step()
            print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
        else:
            adjust_learning_rate(model_optim, epoch + 1, args)
        # adjust_learning_rate(model_optim, epoch + 1, args)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    if args.notrain == 0:
        best_model_path = path + '/' + 'checkpoint.pth'
        model.load_state_dict(torch.load(best_model_path))
    print("------------------------------------")
    mse, mae, smape, mape, nd = test(model, test_data, test_loader, args, device, ii)
    # mse, mae = test(model, vali_data, vali_loader, args, device)
    mses.append(mse)
    maes.append(mae)
    smapes.append(smape)
    mapes.append(mape)
    nds.append(nd)

mses = np.array(mses)
maes = np.array(maes)
mapes = np.array(mapes)
smapes = np.array(smapes)
nds = np.array(nds)
print("mape_mean = {:.4f}, mape_std = {:.4f}".format(np.mean(mapes), np.std(mapes)))
print("smapes_mean = {:.4f}, smapes_std = {:.4f}".format(np.mean(smapes), np.std(smapes)))
print("nds_mean = {:.4f}, nds_std = {:.4f}".format(np.mean(nds), np.std(nds)))