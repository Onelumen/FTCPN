import torch
from torch import nn

class FTCPN(nn.Module):
    def __init__(self, n_inp, n_oup, his_step, n_embding=64, en_layers=4, de_layers=1, proj='linear',
                 activation='relu', maxlevel=1, en_dropout=0.,
                 de_dropout=0., out_split=False, decoder_init_zero=False, bias=False, se_skip=True,
                 attn_conv_params=None):
        super(FTCPN, self).__init__()

        self.args = locals()
        self.n_embding = n_embding
        self.n_oup = n_oup
        self.maxlevel = maxlevel
        self.decoder_init_zero = decoder_init_zero
        self.de_layers = de_layers

        attn_conv_params = {0: {'stride': 2, 'kernel': 2, 'pad': 1}, 1: {'stride': 3, 'kernel': 3, 'pad': 0},
                           2: {'stride': 5, 'kernel': 5, 'pad': 1}} if attn_conv_params == None else attn_conv_params
        if activation == 'relu':
            self.inp_embding = nn.Sequential(Embed(n_inp, n_embding // 2, bias=False, proj=proj), nn.ReLU(),
                                             Embed(n_embding // 2, n_embding, bias=False, proj=proj), nn.ReLU())
        elif activation == 'sigmoid':
            self.inp_embding = nn.Sequential(Embed(n_inp, n_embding // 2, bias=False, proj=proj), nn.Sigmoid(),
                                             Embed(n_embding // 2, n_embding, bias=False, proj=proj), nn.Sigmoid())
        else:
            self.inp_embding = nn.Sequential(Embed(n_inp, n_embding // 2, bias=False, proj=proj), nn.ReLU(),
                                             Embed(n_embding // 2, n_embding, bias=False, proj=proj), nn.ReLU())
        # 输出F1
        self.encoder = nn.LSTM(input_size=n_embding,
                               hidden_size=n_embding,
                               num_layers=en_layers,
                               bidirectional=False,
                               batch_first=True,
                               dropout=en_dropout,
                               bias=False)

        # 轨迹重建模块
        self.trajModule = nn.LSTM(input_size=n_embding,
                                  hidden_size=n_embding,
                                  num_layers=en_layers,
                                  bidirectional=False,
                                  batch_first=True,
                                  dropout=en_dropout,
                                  bias=False)

        # 频域重建模块 TransformerEncoderLayer
        self.freModule = nn.ModuleList([nn.MultiheadAttention(embed_dim=n_embding, num_heads=en_layers)])

    def forward(self, inp):
        """
        :param inp: shape: batch * n_sequence * n_attr
        :param steps: coeff length of wavelet
        :return: coef_set: shape: batch * steps * levels * n_oup,
                 all_scores_set: shape: batch * steps * levels * n_sequence, n_sequence here is timeSteps of inp
        """
        # 输入embdings B*T*6 -> B*T*128
        embdings = self.inp_embding(inp)  # batch * n_sequence * n_embding

        # 输入encoder层，得到
        en_oup, _ = self.encoder(embdings)  # B * T * D

        # 将数据传入轨迹重建模块
        P_1, _ = self.trajModule(en_oup)  # B * T * D

        # 将数据传入频域重建模块
        F_2, _ = self.freModule[0](en_oup, en_oup, en_oup)

        return P_1, F_2


# 这里三个复用即可
class Embed(nn.Module):
    def __init__(self, in_features, out_features, bias=True, proj='linear'):
        super(Embed, self).__init__()
        self.proj = proj
        if proj == 'linear': # 要么简单的线性变换
            self.embed = nn.Linear(in_features, out_features, bias)
        else:                # 要么进行卷积，卷积参数如下
            self.embed = nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=3, stride=1,
                                   padding=1,
                                   padding_mode='replicate', bias=bias)
                                   # 填充采用边界值，而不是全填0
    def forward(self, inp):
        """
        inp: B * T * D
        """
        if self.proj == 'linear':
            inp = self.embed(inp)
        else:
            inp = self.embed(inp.transpose(1, 2)).transpose(1, 2)
        return inp          # 卷积要求，两次变形

class WaveletAttention(nn.Module):
    def __init__(self, hid_dim, init_steps, skip=True, kernel=3, stride=3, padding=0):
        super(WaveletAttention, self).__init__()
        self.enhance = EnhancedBlock(hid_size=hid_dim, channel=init_steps, skip=skip)
        self.convs = nn.Sequential(nn.Conv1d(in_channels=hid_dim,
                                             out_channels=hid_dim,
                                             kernel_size=kernel,
                                             stride=stride, padding=padding,
                                             padding_mode='zeros',
                                             bias=False),
                                   nn.ReLU())

    def forward(self, hid_set: torch.Tensor):
        """
        q = W1 * hid_0, K = W2 * hid_set
        softmax(q * K.T) * hid_set
        :param hid_set: Key set, shape: batch * n_sequence * hid_dim
        :param hid_0: Query hidden state, shape: batch * 1 * hid_dim
        :return: out: shape: shape: batch * 1 * hid_dim, scores: shape: batch * 1 * sequence
        """
        reweighted, weight = self.enhance(hid_set)
        align = self.convs(reweighted.transpose(1, 2)).transpose(1, 2)  # B * DestStep * D
        return align, weight


class EnhancedBlock(nn.Module):
    def __init__(self, hid_size, channel, skip=True):
        super(EnhancedBlock, self).__init__()
        self.skip = skip
        self.comp = nn.Sequential(nn.Linear(hid_size, hid_size // 2, bias=False),
                                  nn.ReLU(),
                                  nn.Linear(hid_size // 2, 1, bias=False))

        self.activate = nn.Sequential(nn.Linear(channel, channel // 2, bias=False),
                                      nn.ReLU(),
                                      nn.Linear(channel // 2, channel, bias=False),
                                      nn.Sigmoid())

    def forward(self, inp):
        S = self.comp(inp)  # B * T * 1
        E = self.activate(S.transpose(1, 2))  # B * 1 * T
        out = inp * E.transpose(1, 2).expand_as(inp)
        if self.skip:
            out += inp
        return out, E  # B * T * D