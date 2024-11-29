import os
import time
import numpy as np
import matplotlib.pyplot as plt
import math
import shutil
import torch
from model import WTFTP, WTFTP_AttnRemoved
from torch.utils.data import DataLoader
from dataloader import DataGenerator
from utils import progress_bar, wt_coef_len, recorder
import argparse
import logging
import datetime
from tensorboardX import SummaryWriter
from pytorch_wavelets import DWT1DForward, DWT1DInverse


class Train:
    def __init__(self, opt, SEED=None, tracking=False):
        if SEED is not None:
            torch.manual_seed(SEED)
        if tracking:
            self.dev_recorder = recorder('dev_loss')
            self.train_recorder = recorder('train_loss', 'temporal_loss', 'freq_loss')

        self.SEED = SEED
        self.opt = opt
        self.iscuda = torch.cuda.is_available()
        self.device = torch.device(f'cuda:{self.opt.cuda}' if self.iscuda and not opt.cpu else 'cpu')
        # 1. 加载数据
        self.data_set = DataGenerator(data_path=self.opt.datadir,
                                      minibatch_len=opt.minibatch_len, interval=opt.interval,
                                      use_preset_data_ranges=False)

        # 2. 根据设置--是否含有注意力模块，而产生的两个网络架构module，且网络架构已经写在model.py中

        if self.opt.attn: # 定义了net网络
            self.net = WTFTP(
                n_inp=6,
                n_oup=6,
                his_step=opt.minibatch_len - 1, # 步长是10-1=9
                n_embding=opt.embding, # 扩大到64
                en_layers=opt.enlayer, # encoder的lstm，选择了4
                de_layers=opt.delayer, # decoder的lstm，选择了1
                activation='relu',
                proj='linear',
                maxlevel=opt.maxlevel,
                en_dropout=opt.dpot,
                de_dropout=opt.dpot,
                bias=True
            )
        else:
            self.net = WTFTP_AttnRemoved(n_inp=6,
                                         n_oup=6,
                                         n_embding=opt.embding,
                                         n_encoderLayers=opt.enlayer,
                                         n_decoderLayers=opt.delayer,
                                         activation='relu',
                                         proj='linear',
                                         maxlevel=opt.maxlevel,
                                         en_dropout=opt.dpot,
                                         de_dropout=opt.dpot)

        # 3. 设置损失函数
        self.MSE = torch.nn.MSELoss(reduction='mean') #意为求平均的MSE
        self.MAE = torch.nn.L1Loss(reduction='mean') #意为求平均的MAE

        # 4. 设置优化器，这里的net是上面定义好的net是WTFTP
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.opt.lr)
        self.opt_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)

        # 5. 处理目录以及Tensorboard的问题
        if not self.opt.nologging:
            self.log_path = self.opt.logdir + f'/{datetime.datetime.now().strftime("%y-%m-%d")}'
            if self.opt.debug:
                self.log_path = self.opt.logdir + f'/DEBUG-{datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")}{"-" + self.opt.comments if len(self.opt.comments) > 0 else ""}'
                self.opt.saving_model_num = 10
            self.TX_log_path = self.log_path + '/Tensorboard'
            if not os.path.exists(self.log_path):
                os.makedirs(self.log_path)
            if os.path.exists(self.TX_log_path):
                shutil.rmtree(self.TX_log_path)
                os.mkdir(self.TX_log_path)
            else:
                os.mkdir(self.TX_log_path)
            logging.basicConfig(filename=os.path.join(self.log_path, 'train.log'),
                                filemode='a', format='%(asctime)s   %(message)s', level=logging.DEBUG)
            self.TX_logger = SummaryWriter(log_dir=self.TX_log_path) #这里的path是log/24-05-14/Tensorboard
        if self.opt.saving_model_num > 0:
            self.model_names = []

    def train(self):
        self.net.to(self.device)
        self.net.args['train_opt'] = self.opt
        if not self.opt.nologging:
            log_str = f'TRAIN DETAILS'.center(50, '-') + \
                      f'\ntraining on device {self.device}...\n'
            for arg in vars(self.opt):
                log_str += f'{arg}: {getattr(self.opt, arg)}\n'
            log_str += f'MODEL DETAILS'.center(50, '-') + '\n'
            for key in self.net.args:
                log_str += f'{key}: {self.net.args[key]}\n'
            logging.debug(10 * '\n' + f'Beginning of the training epoch'.center(50, '-'))
            logging.debug(log_str)
            print(log_str)

        for i in range(self.opt.epoch): # 针对一次训练i,i从0开始，所以需要在下面训练调用时+1
            print('\n' + f'Epoch {i+1}/{self.opt.epoch}'.center(50, '-'))
            if not self.opt.nologging:
                logging.debug('\n' + f'Epoch {i+1}/{self.opt.epoch}'.center(50, '-'))
            print(f'lr: {float(self.opt_lr_scheduler.get_last_lr()[0])}')
            self.train_each_epoch(i+1) # 此处训练调用+1
            self.saving_model(self.log_path, f'epoch_{i+1}.pt') # 保持此次epoch训练结果
            self.opt_lr_scheduler.step() # 优化器调用一次
            self.dev_each_epoch(i+1)  # 交叉验证集训练一次
            # 没有看到早停或超参数调整呢？

        if not self.opt.nologging:
            logging.debug(10 * '\n' + f'End of the training epoch'.center(50, '-'))
            if hasattr(self, 'dev_recorder'):
                self.dev_recorder.push(self.log_path, 'dev_recorder.pt')
            if hasattr(self, 'train_recorder'):
                self.train_recorder.push(self.log_path, 'train_recorder.pt')

    def train_each_epoch(self, epoch):
        # 加载数据
        train_data = DataLoader(dataset=self.data_set.train_set, batch_size=self.opt.batch_size, shuffle=True,
                                collate_fn=self.data_set.collate, num_workers=0, pin_memory=self.iscuda)
        # 设置训练状态
        self.net.train()

        # 创建直接可以使用的dwt和idwt模块
        dwt = DWT1DForward(wave=self.opt.wavelet, J=self.opt.maxlevel, mode=self.opt.wt_mode).to(self.device)
        idwt = DWT1DInverse(wave=self.opt.wavelet, mode=self.opt.wt_mode).to(self.device)


        start_time = time.perf_counter()
        batchs_len = len(train_data)

        # 设计损失记录list
        train_loss_set = []
        temporal_loss_set = []
        freq_loss_set = []

        for i, batch in enumerate(train_data):
            # 准备数据，放到gpu里面
            batch = torch.FloatTensor(batch).to(self.device)

            inp_batch = batch[:, :-1, :]  # shape: batch * n_sequence * n_attr
            train_wt_loss = torch.tensor(0.0).to(self.device)
            tgt_batch = batch  # use the all to calculate wavelet coefficients, shape: batch * n_sequence * n_attr

            # contiguous保证张量放到内存中是连续的
            wt_tgt_batch = dwt(
                batch.transpose(1, 2).contiguous())  # tuple(lo, hi), shape: batch * n_attr * n_sequence

            # 在这里处理数据


            # 把不含标签的数据送入net，得到的是频域特征，但是输入的是直接的变量
            if self.opt.attn:
                wt_pre_batch, score_set = self.net(inp_batch) # net是已经定义好的网络
                # 这个注意力分数最后怎么用呢？
            else:
                wt_pre_batch = self.net(inp_batch)

            # 逆离散小波变换，把分量变成原来数据
            pre_batch = idwt((wt_pre_batch[-1].transpose(1, 2).contiguous(),
                              [comp.transpose(1, 2).contiguous() for comp in
                               0[:-1]])).contiguous()  # shape: batch * n_attr * n_sequence
            pre_batch = pre_batch.transpose(1, 2)  # shape: batch * n_sequence * n_attr

            # 计算损失:问题是原序列直接进入了net，并没有经过dwt模块，那怎么做频域Loss呢？
            train_temporal_loss, loss_details = self.cal_spatial_loss(tgt_batch, pre_batch)# pre_batch甚至是经过idwt了
            train_wt_loss = self.cal_freq_loss(wt_tgt_batch, wt_pre_batch)

            # 给反向传播设置公式
            loss_backward = torch.tensor(self.opt.w_spatial, dtype=torch.float,
                                         device=self.device) * train_temporal_loss + train_wt_loss

            # 优化器工作
            self.optimizer.zero_grad()
            loss_backward.backward()
            self.optimizer.step()

            with torch.no_grad():
                # 记录并且存储loss
                train_loss_set.append(float(loss_backward.detach().cpu()))
                temporal_loss_set.append(float(train_temporal_loss.detach().cpu()))
                freq_loss_set.append(float(train_wt_loss.detach().cpu()))

            # print loss
            print_str = f'train_loss(scaled): {loss_backward.item():.8f}, temporal_loss: {train_temporal_loss.item():.8f}, ' \
                        f'freq_loss: {train_wt_loss.item():.8f}'
            progress_bar(i, batchs_len, print_str, start_time) # 进度条设置
            record_freq = 20
            if not self.opt.nologging and (i % ((batchs_len - 1) // record_freq) == 0 or i == batchs_len - 1):
                logging.debug(f'{i}/{batchs_len - 1} ' + print_str)
        print_str = f'ave_train_loss: {np.mean(train_loss_set):.8f}, ave_temporal_loss: {np.mean(temporal_loss_set):.8f}' \
                    + f', ave_freq_loss: {np.mean(freq_loss_set):.8f}'
        print(print_str)

        # 打印到TensorBoard
        if not self.opt.nologging:
            logging.debug(print_str)
            self.TX_logger.add_scalar('train_loss', np.mean(train_loss_set), global_step=epoch)
            self.TX_logger.add_scalar('train_temporal_loss', np.mean(temporal_loss_set), global_step=epoch)
            self.TX_logger.add_scalar('train_freq_loss', np.mean(freq_loss_set), global_step=epoch)
        if hasattr(self, 'train_recorder'):
            self.train_recorder.add('train_loss', np.mean(train_loss_set))
            self.train_recorder.add('temporal_loss', np.mean(temporal_loss_set))
            self.train_recorder.add('freq_loss', np.mean(freq_loss_set))
        if not self.opt.nologging and self.opt.attn:
            project = plt.cm.get_cmap('YlGnBu')
            with torch.no_grad():
                self.TX_logger.add_image(f'score',
                                         project(torch.mean(score_set.clone().cpu(), dim=0).numpy())[:, :, :-1],
                                         dataformats='HWC',
                                         global_step=epoch)

    def dev_each_epoch(self, epoch):
        dev_data = DataLoader(dataset=self.data_set.dev_set, batch_size=self.opt.batch_size, shuffle=False,
                              collate_fn=self.data_set.collate, num_workers=0, pin_memory=self.iscuda)
        self.net.eval()
        idwt = DWT1DInverse(wave=self.opt.wavelet, mode=self.opt.wt_mode).to(self.device)

        tgt_set = []
        pre_set = []

        wt_coef_length = wt_coef_len(self.opt.minibatch_len, wavelet=self.opt.wavelet, mode=self.opt.wt_mode,
                                     maxlevel=self.opt.maxlevel)
        with torch.no_grad():

            for i, batch in enumerate(dev_data):
                batch = torch.FloatTensor(batch).to(self.device)
                n_batch, _, n_attr = batch.shape
                inp_batch = batch[:, :-1, :]  # shape: batch * n_sequence * n_attr
                tgt_batch = batch
                if self.opt.attn:
                    wt_pre_batch, score_set = self.net(inp_batch)
                else:
                    wt_pre_batch = self.net(inp_batch)
                pre_batch = idwt((wt_pre_batch[-1].transpose(1, 2).contiguous(),
                                  [comp.transpose(1, 2).contiguous() for comp in
                                   wt_pre_batch[:-1]])).contiguous()  # shape: batch * n_attr * n_sequence
                pre_batch = pre_batch.transpose(1, 2)  # shape: batch * n_sequence * n_attr
                tgt_set.append(tgt_batch)
                pre_set.append(pre_batch)
            tgt_set = torch.cat(tgt_set, dim=0)
            pre_set = torch.cat(pre_set, dim=0)
            dev_loss, loss_details = self.cal_spatial_loss(tgt_set, pre_set, is_training=False)
        print_str = f'Evaluation-Stage:\n' \
                    f'aveMSE(scaled): {dev_loss:.8f}, in each attr(RMSE, unscaled): {loss_details["rmse"]}\n' \
                    f'aveMAE(scaled): None, in each attr(MAE, unscaled): {loss_details["mae"]}'
        print(print_str)
        if not self.opt.nologging:
            logging.debug(print_str)
            self.TX_logger.add_scalar('eval_aveMSE', dev_loss, global_step=epoch)
        if hasattr(self, 'dev_recorder'):
            self.dev_recorder.add('dev_loss', dev_loss)

    def cal_spatial_loss(self, tgt, pre, is_training=True):
        """
        :param tgt: shape: batch * n_sequence * n_attr
        :param pre: shape: batch * n_sequence+? * n_attr
        :return:
        """
        n_sequence = tgt.shape[1]

        # 针对训练/预测 来设置权重矩阵
        last_node_weight = 1.0
        if is_training:
            weights = torch.ones(n_sequence, dtype=torch.float, device=tgt.device)
        else:
            weights = torch.zeros(n_sequence, dtype=torch.float, device=tgt.device)
        weights[-1] = last_node_weight
        weighted_loss = torch.tensor(0.0).to(self.device)

        # 计算（理解：在时间尺度上我们只关心最后一个时间步的MSE即可
        for i in range(n_sequence):
            weighted_loss += weights[i] * self.MSE(pre[:, i, :], tgt[:, i, :])
        try:
            weighted_loss /= weights.count_nonzero()
        except:
            if is_training:
                weighted_loss /= torch.tensor(n_sequence, dtype=torch.float).to(self.device)
            else:
                weighted_loss /= torch.tensor(1.0, dtype=torch.float).to(self.device)

        # 还原无尺度：即原始数据
        loss_unscaled = {'rmse': {}, 'mae': {}}
        # clone出数据，避免反向传播训练影响
        tgt_cloned = tgt.detach()
        pre_cloned = pre.detach()

        for i, name in enumerate(self.data_set.attr_names): # 针对每一个属性在尺度上的数据
            loss_unscaled['rmse'][name] = math.sqrt(
                # 注意：此处unscale还原尺度了
                float(self.MSE(self.data_set.unscale(tgt_cloned[:, n_sequence - 1, i], name),
                               self.data_set.unscale(pre_cloned[:, n_sequence - 1, i], name))))
            loss_unscaled['mae'][name] = float(self.MAE(self.data_set.unscale(tgt_cloned[:, n_sequence - 1, i], name),
                                                        self.data_set.unscale(pre_cloned[:, n_sequence - 1, i], name)))
        return weighted_loss, loss_unscaled

    def cal_freq_loss(self, tgt, pre) -> torch.Tensor:
        """
        :param tgt: tuple:(lo, hi) shape: batch * n_attr * n_sequence
        :param pre: list:[hi's lo] shape: batch * n_attr * n_sequence
        :return:
        """
        wt_loss = torch.tensor(0.0).to(self.device) # 先创建在gpu上面的损失分量
        # opt.w_lo默认为1.0,w_hi也默认为1,这两个是权重乘在结果前面，maxlevel是指定的
        wt_loss += self.opt.w_lo * self.MSE(tgt[0], pre[-1].transpose(1, 2))
        for i in range(self.opt.maxlevel): # 在不同层级的小波分量上分别计算并且求和
            wt_loss += self.opt.w_hi * self.MSE(pre[i].transpose(1, 2), tgt[1][i])
            #pre[i]是第i层
        # 分层权重求
        return wt_loss


    # 保存模型的函数
    def saving_model(self, model_path, this_model_name):
        if self.opt.saving_model_num > 0:
            self.model_names.append(this_model_name)
            if len(self.model_names) > self.opt.saving_model_num:
                removed_model_name = self.model_names[0]
                del self.model_names[0]
                os.remove(os.path.join(model_path, removed_model_name))  # remove the oldest model
            torch.save(self.net, os.path.join(model_path, this_model_name))  # save the latest model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--minibatch_len', default=10, type=int)
    parser.add_argument('--interval', default=1, type=int)
    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--epoch', default=150, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--dpot', default=0.0, type=float)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--nologging', action='store_true') # 开关
    parser.add_argument('--logdir', default='./log', type=str)
    parser.add_argument('--datadir', default='./data', type=str)
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--saving_model_num', default=0, type=int)
    parser.add_argument('--debug', action='store_true',
                        help='logs saving in an independent dir and 10 models being saved')
    parser.add_argument('--maxlevel', default=1, type=int)
    parser.add_argument('--wavelet', default='haar', type=str)
    parser.add_argument('--L2details', action='store_true',
                        help='L2 regularization for detail coefficients to avoid them to converge to zeros')
    parser.add_argument('--wt_mode', default='symmetric', type=str)
    parser.add_argument('--w_spatial', default='0.0', type=float)
    parser.add_argument('--w_lo', default='1.0', type=float)
    parser.add_argument('--w_hi', default='1.0', type=float)
    parser.add_argument('--enlayer', default='4', type=int)
    parser.add_argument('--delayer', default='1', type=int)
    parser.add_argument('--embding', default='64', type=int)
    parser.add_argument('--attn', action='store_true') #action这里是开关设置
    parser.add_argument('--comments', default="", type=str,
                        help='comments in the dir name for identification')
    parser.add_argument('--cuda', default=0, type=int)
    args = parser.parse_args()
    # 添加参数，创建类，开始训练
    train = Train(args, tracking=True)
    train.train()