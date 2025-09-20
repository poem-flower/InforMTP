from data.data_loader import My_Dataset
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np
import scipy.io as scio

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

import warnings
warnings.filterwarnings('ignore')

class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'informer':Informer,
            'informerstack':InformerStack,
        }
        if self.args.model=='informer' or self.args.model=='informerstack':
            e_layers = self.args.e_layers if self.args.model=='informer' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers, # self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args
        Data = My_Dataset
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
            args.data_path = 'my_data_test.mat'
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            args.data_path = 'my_data_track.mat'
        elif flag=='train':
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
            args.data_path = 'my_data_train.mat'
        elif flag == 'val':
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size;freq = args.freq
            args.data_path = 'my_data_val.mat'

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,seq_true) in enumerate(vali_loader):
            pred, true = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = criterion(pred.detach().cpu(), seq_true.float().detach().cpu())
            total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        loss_output = []

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y,seq_true) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y)
                loss = criterion(pred, seq_true.float().to(self.device))
                train_loss.append(loss.item())
                loss_output.append(loss.item())
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            data_mode = self.args.root_path.removeprefix('./data/')
            early_stopping(vali_loss, self.model, path+'/'+data_mode)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)

        data_mode = self.args.root_path.removeprefix('./data/')
        best_model_path = path+'/'+data_mode+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        folder_path = './matlab/network_output/'
        scio.savemat(folder_path + data_mode + 'loss.mat', {'loss': loss_output})
        
        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []  # preds为网络输出结果，即滤波和预测序列，长度label_len + pred_len
        trues = []  # true为与pred做loss的理想输出结果
        x = []  # x为编码器输入，即量测差分，长度seq_len
        
        for i, (batch_x, batch_y, seq_true) in enumerate(test_loader):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y)
            preds.append(pred.detach().cpu().numpy())
            trues.append(seq_true.detach().cpu().numpy())
            x.append(batch_x.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        x = np.array(x)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        x = x.reshape(-1, x.shape[-2], x.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        data_mode = self.args.root_path.removeprefix('./data/')
        folder_path = './matlab/network_output/' + data_mode
        scio.savemat(folder_path + 'true.mat', {'true': trues})
        scio.savemat(folder_path + 'pred.mat', {'pred': preds})
        scio.savemat(folder_path + 'x_true.mat', {'x_true': x})

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            data_mode = self.args.root_path.removeprefix('./data/')
            best_model_path = path + '/' + data_mode + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        preds = []

        for i, (batch_x, batch_y, seq_true) in enumerate(pred_loader):
            pred, true = self._process_one_batch(pred_data, batch_x, batch_y)
            pred = pred.detach().cpu().numpy()

            preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(pred_data.M, pred_data.n, preds.shape[-2], preds.shape[-1])

        data_mode = self.args.root_path.removeprefix('./data/')
        folder_path = './matlab/network_output/' + data_mode
        scio.savemat(folder_path + 'tracked.mat', {'tracked': preds})

        return


    def _process_one_batch(self, dataset_object, batch_x, batch_y):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_y,)[0]
                else:
                    outputs = self.model(batch_x, batch_y)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_y)[0]
            else:
                outputs = self.model(batch_x, batch_y)
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-(self.args.pred_len+self.args.label_len):,f_dim:].to(self.device)

        return outputs, batch_y

