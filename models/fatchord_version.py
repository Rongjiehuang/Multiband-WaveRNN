import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.distribution import sample_from_discretized_mix_logistic
from utils.display import *
from utils.dsp import *
import os
import numpy as np
from pathlib import Path
from typing import Union
from models.pqmf import PQMF


class ResBlock(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.conv1 = nn.Conv1d(dims, dims, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(dims, dims, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(dims)
        self.batch_norm2 = nn.BatchNorm1d(dims)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        return x + residual


class MelResNet(nn.Module):
    def __init__(self, res_blocks, in_dims, compute_dims, res_out_dims, pad):
        super().__init__()
        k_size = pad * 2 + 1
        self.conv_in = nn.Conv1d(in_dims, compute_dims, kernel_size=k_size, bias=False)
        self.batch_norm = nn.BatchNorm1d(compute_dims)
        self.layers = nn.ModuleList()
        for i in range(res_blocks):
            self.layers.append(ResBlock(compute_dims))
        self.conv_out = nn.Conv1d(compute_dims, res_out_dims, kernel_size=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        for f in self.layers: x = f(x)
        x = self.conv_out(x)
        return x


class Stretch2d(nn.Module):
    def __init__(self, x_scale, y_scale):
        super().__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.unsqueeze(-1).unsqueeze(3)
        x = x.repeat(1, 1, 1, self.y_scale, 1, self.x_scale)  # 通过复制的方式上采样
        return x.view(b, c, h * self.y_scale, w * self.x_scale)  # 扩大特征向量


class UpsampleNetwork(nn.Module):
    def __init__(self, feat_dims, upsample_scales, compute_dims,
                 res_blocks, res_out_dims, pad):
        super().__init__()
        total_scale = np.cumproduct(upsample_scales)[-1]
        self.indent = pad * total_scale
        self.resnet = MelResNet(res_blocks, feat_dims, compute_dims, res_out_dims, pad)
        self.resnet_stretch = Stretch2d(total_scale, 1)
        self.up_layers = nn.ModuleList()
        for scale in upsample_scales:
            k_size = (1, scale * 2 + 1)
            padding = (0, scale)
            stretch = Stretch2d(scale, 1)
            conv = nn.Conv2d(1, 1, kernel_size=k_size, padding=padding, bias=False)
            conv.weight.data.fill_(1. / k_size[1])
            self.up_layers.append(stretch)
            self.up_layers.append(conv)

    def forward(self, m):
        aux = self.resnet(m).unsqueeze(1)
        aux = self.resnet_stretch(aux)
        aux = aux.squeeze(1)
        m = m.unsqueeze(1)
        for f in self.up_layers: m = f(m)
        m = m.squeeze(1)[:, :, self.indent:-self.indent]
        return m.transpose(1, 2), aux.transpose(1, 2)


class WaveRNN(nn.Module):
    def __init__(self, rnn_dims, fc_dims, bits, pad, upsample_factors,
                 feat_dims, compute_dims, res_out_dims, res_blocks,
                 hop_length, sample_rate, pad_val, mode='RAW'):
        super().__init__()
        self.mode = mode
        self.pad = pad
        self.pad_val = pad_val
        if self.mode == 'RAW':
            self.n_classes = 2 ** bits
        elif self.mode == 'MOL':
            self.n_classes = 30
        else:
            RuntimeError("Unknown model mode value - ", self.mode)

        # List of rnns to call `flatten_parameters()` on
        self._to_flatten = []

        self.rnn_dims = rnn_dims
        self.aux_dims = res_out_dims // 4
        self.hop_length = hop_length
        self.sample_rate = sample_rate

        self.upsample = UpsampleNetwork(feat_dims, upsample_factors, compute_dims, res_blocks, res_out_dims, pad)
        self.I = nn.Linear(feat_dims + self.aux_dims + 4, rnn_dims)

        self.rnn10 = nn.GRU(rnn_dims, rnn_dims, batch_first=True)
        self.rnn11 = nn.GRU(rnn_dims, rnn_dims, batch_first=True)
        self.rnn12 = nn.GRU(rnn_dims, rnn_dims, batch_first=True)
        self.rnn13 = nn.GRU(rnn_dims, rnn_dims, batch_first=True)
        self.rnn20 = nn.GRU(rnn_dims + self.aux_dims, rnn_dims, batch_first=True)
        self.rnn21 = nn.GRU(rnn_dims + self.aux_dims, rnn_dims, batch_first=True)
        self.rnn22 = nn.GRU(rnn_dims + self.aux_dims, rnn_dims, batch_first=True)
        self.rnn23 = nn.GRU(rnn_dims + self.aux_dims, rnn_dims, batch_first=True)

        # self._to_flatten += [self.rnn1, self.rnn2]
        self._to_flatten += [self.rnn10, self.rnn11,self.rnn12, self.rnn13,self.rnn20, self.rnn21,self.rnn22, self.rnn23]

        self.fc10 = nn.Linear(rnn_dims + self.aux_dims, fc_dims)
        self.fc11 = nn.Linear(rnn_dims + self.aux_dims, fc_dims)
        self.fc12 = nn.Linear(rnn_dims + self.aux_dims, fc_dims)
        self.fc13 = nn.Linear(rnn_dims + self.aux_dims, fc_dims)

        self.fc20 = nn.Linear(fc_dims + self.aux_dims, fc_dims)
        self.fc21 = nn.Linear(fc_dims + self.aux_dims, fc_dims)
        self.fc22 = nn.Linear(fc_dims + self.aux_dims, fc_dims)
        self.fc23 = nn.Linear(fc_dims + self.aux_dims, fc_dims)

        self.fc31 = nn.Linear(fc_dims, self.n_classes)
        self.fc32 = nn.Linear(fc_dims, self.n_classes)
        self.fc33 = nn.Linear(fc_dims, self.n_classes)
        self.fc30 = nn.Linear(fc_dims, self.n_classes)

        self.register_buffer('step', torch.zeros(1, dtype=torch.long))
        self.num_params()

        # Avoid fragmentation of RNN parameters and associated warning
        self._flatten_parameters()

    def forward(self, x, mels):  # x: (Batch, Subband, T)
        device = next(self.parameters()).device  # use same device as parameters

        # Although we `_flatten_parameters()` on init, when using DataParallel
        # the model gets replicated, making it no longer guaranteed that the
        # weights are contiguous in GPU memory. Hence, we must call it again
        self._flatten_parameters()

        self.step += 1
        bsize = x.size(0)
        # h1 = torch.zeros(1, bsize, self.rnn_dims, device=device)
        # h2 = torch.zeros(1, bsize, self.rnn_dims, device=device)
        mels, aux = self.upsample(mels)

        aux_idx = [self.aux_dims * i for i in range(5)]
        a1 = aux[:, :, aux_idx[0]:aux_idx[1]]
        a2 = aux[:, :, aux_idx[1]:aux_idx[2]]
        a3 = aux[:, :, aux_idx[2]:aux_idx[3]]
        a4 = aux[:, :, aux_idx[3]:aux_idx[4]]
        # x = torch.cat([x.unsqueeze(-1), mels, a1], dim=2)

        x = torch.cat([x.transpose(1,2), mels, a1], dim=2)  # (batch,T,4)  (batch,T,80) (batch,T,32)
        x = self.I(x)  # (batch, T, 116) -> # (batch, T, 512)
        res = x
        # x, _ = self.rnn1(x, h1)

        x0, _ = self.rnn10(x)   # 不加入隐藏层-Begee  # (batch, T, 512) -> (batch, T, 512)
        x1, _ = self.rnn11(x)  # 不加入隐藏层-Begee  # (batch, T, 512) -> (batch, T, 512)
        x2, _ = self.rnn12(x)  # 不加入隐藏层-Begee  # (batch, T, 512) -> (batch, T, 512)
        x3, _ = self.rnn13(x)  # 不加入隐藏层-Begee  # (batch, T, 512) -> (batch, T, 512)

        x0 = x0 + res    # (batch, T, 512)
        x1 = x1 + res    # (batch, T, 512)
        x2 = x2 + res    # (batch, T, 512)
        x3 = x3 + res    # (batch, T, 512)

        res0 = x0
        res1 = x1
        res2 = x2
        res3 = x3

        x0 = torch.cat([x0, a2], dim=2)  # (batch, T, 512) -> (batch, T, 512+128)
        x1 = torch.cat([x1, a2], dim=2)  # (batch, T, 512) -> (batch, T, 512+128)
        x2 = torch.cat([x2, a2], dim=2)  # (batch, T, 512) -> (batch, T, 512+128)
        x3 = torch.cat([x3, a2], dim=2)  # (batch, T, 512) -> (batch, T, 512+128)

        x0, _ = self.rnn20(x0)  # 不加入隐藏层-Begee  (batch, T, 512+128) -> (batch, T, 512)
        x1, _ = self.rnn21(x1)  # 不加入隐藏层-Begee  (batch, T, 512+128) -> (batch, T, 512)
        x2, _ = self.rnn22(x2)  # 不加入隐藏层-Begee  (batch, T, 512+128) -> (batch, T, 512)
        x3, _ = self.rnn23(x3)  # 不加入隐藏层-Begee  (batch, T, 512+128) -> (batch, T, 512)

        x0 = x0 + res0
        x1 = x1 + res1
        x2 = x2 + res2
        x3 = x3 + res3

        x0 = torch.cat([x0, a3], dim=2)  # (batch, T, 512+128)
        x1 = torch.cat([x1, a3], dim=2)  # (batch, T, 512+128)
        x2 = torch.cat([x2, a3], dim=2)  # (batch, T, 512+128)
        x3 = torch.cat([x3, a3], dim=2) # (batch, T, 512+128)

        x0 = F.relu(self.fc10(x0)) # (batch, T, 512+128) -> (batch, T, 512)
        x1 = F.relu(self.fc11(x1))  # (batch, T, 512+128) -> (batch, T, 512)
        x2 = F.relu(self.fc12(x2))  # (batch, T, 512+128) -> (batch, T, 512)
        x3 = F.relu(self.fc13(x3))  # (batch, T, 512+128) -> (batch, T, 512)

        x0 = torch.cat([x0, a4], dim=2)  # (batch, T, 512+128)
        x1 = torch.cat([x1, a4], dim=2)  # (batch, T, 512+128)
        x2 = torch.cat([x2, a4], dim=2)  # (batch, T, 512+128)
        x3 = torch.cat([x3, a4], dim=2)  # (batch, T, 512+128)

        x0 = F.relu(self.fc20(x0))  # (batch, T, 512+128) -> (batch, T, 512)
        x1 = F.relu(self.fc21(x1))  # (batch, T, 512+128) -> (batch, T, 512)
        x2 = F.relu(self.fc22(x2))  # (batch, T, 512+128) -> (batch, T, 512)
        x3 = F.relu(self.fc23(x3))  # (batch, T, 512+128) -> (batch, T, 512)

        out0 = self.fc30(x0).unsqueeze(-1) # (batch, T, 512) -> (batch, T, 512, 1)
        out1 = self.fc31(x1).unsqueeze(-1)
        out2 = self.fc32(x2).unsqueeze(-1)
        out3 = self.fc33(x3).unsqueeze(-1)
        out = torch.cat([out0,out1,out2,out3], dim=3)  # (B, T, num_classes, sub_band)
        return out


    def generate(self, mels, save_path: Union[str, Path],save_path2: Union[str, Path], batched, target, overlap, mu_law):
        self.eval()

        device = next(self.parameters()).device  # use same device as parameters

        mu_law = mu_law if self.mode == 'RAW' else False

        output = []
        start = time.time()
        rnn10 = self.get_gru_cell(self.rnn10)
        rnn11 = self.get_gru_cell(self.rnn11)
        rnn12 = self.get_gru_cell(self.rnn12)
        rnn13 = self.get_gru_cell(self.rnn13)
        rnn20 = self.get_gru_cell(self.rnn20)
        rnn21 = self.get_gru_cell(self.rnn21)
        rnn22 = self.get_gru_cell(self.rnn22)
        rnn23 = self.get_gru_cell(self.rnn23)

        mypqmf = PQMF()
        with torch.no_grad():
            #   MB-WaveRNN    |     WaveRNN
            mels = torch.as_tensor(mels, device=device)  # (80, 748)
            wave_len = (mels.size(-1) - 1) * self.hop_length
            mels = self.pad_tensor(mels.transpose(1, 2), self.pad, self.pad_val,side='both')  # (752, 80)
            mels, aux = self.upsample(mels.transpose(1, 2))  # (23936,80) (23936,128) | (95744,80) (95744,128)
            # print("mels.shape",mels.shape,"aux.shape",aux.shape)
            if batched:
                mels = self.fold_with_overlap(mels, target, overlap,self.pad_val)
                aux = self.fold_with_overlap(aux, target, overlap,self.pad_val)

            b_size, seq_len, _ = mels.size()

            h10 = torch.zeros(b_size, self.rnn_dims, device=device)
            h11 = torch.zeros(b_size, self.rnn_dims, device=device)
            h12 = torch.zeros(b_size, self.rnn_dims, device=device)
            h13 = torch.zeros(b_size, self.rnn_dims, device=device)
            h20 = torch.zeros(b_size, self.rnn_dims, device=device)
            h21 = torch.zeros(b_size, self.rnn_dims, device=device)
            h22 = torch.zeros(b_size, self.rnn_dims, device=device)
            h23 = torch.zeros(b_size, self.rnn_dims, device=device)

            x = torch.zeros(b_size, 4, device=device)

            d = self.aux_dims
            aux_split = [aux[:, :, d * i:d * (i + 1)] for i in range(4)]

            #########################  MultiBand-WaveRNN   #########################
            if hp.voc_multiband:
                for i in range(seq_len):  # 23936 | 95744

                    m_t = mels[:, i, :]
                    a1_t, a2_t, a3_t, a4_t = \
                            (a[:, i, :] for a in aux_split)

                    # print("x.shape",x.shape,"m_t.shape",m_t.shape,"a1_t.shape",a1_t.shape)
                    x = torch.cat([x, m_t, a1_t], dim=1)  #(5,4) + (5,32) + (5,80)
                    x = self.I(x)

                    h10 = rnn10(x, h10)
                    h11 = rnn11(x, h11)
                    h12 = rnn12(x, h12)
                    h13 = rnn13(x, h13)

                    x0 = x + h10
                    x1 = x + h11
                    x2 = x + h12
                    x3 = x + h13

                    inp0 = torch.cat([x0, a2_t], dim=1)
                    inp1 = torch.cat([x1, a2_t], dim=1)
                    inp2 = torch.cat([x2, a2_t], dim=1)
                    inp3 = torch.cat([x3, a2_t], dim=1)

                    h20 = rnn20(inp0, h20)
                    h21 = rnn21(inp1, h21)
                    h22 = rnn22(inp2, h22)
                    h23 = rnn23(inp3, h23)

                    x0 = x0 + h20
                    x1 = x1 + h21
                    x2 = x2 + h22
                    x3 = x3 + h23

                    x0 = torch.cat([x0, a3_t], dim=1)
                    x1 = torch.cat([x1, a3_t], dim=1)
                    x2 = torch.cat([x2, a3_t], dim=1)
                    x3 = torch.cat([x3, a3_t], dim=1)

                    x0 = F.relu(self.fc10(x0))
                    x1 = F.relu(self.fc11(x1))
                    x2 = F.relu(self.fc12(x2))
                    x3 = F.relu(self.fc13(x3))

                    x0 = torch.cat([x0, a4_t], dim=1)
                    x1 = torch.cat([x1, a4_t], dim=1)
                    x2 = torch.cat([x2, a4_t], dim=1)
                    x3 = torch.cat([x3, a4_t], dim=1)

                    x0 = F.relu(self.fc20(x0))
                    x1 = F.relu(self.fc21(x1))
                    x2 = F.relu(self.fc22(x2))
                    x3 = F.relu(self.fc23(x3))

                    logits0 = self.fc30(x0)
                    logits1 = self.fc31(x1)
                    logits2 = self.fc32(x2)
                    logits3 = self.fc33(x3)

                    if self.mode == 'MOL':
                        sample0 = sample_from_discretized_mix_logistic(logits0.unsqueeze(0).transpose(1, 2))
                        sample1 = sample_from_discretized_mix_logistic(logits1.unsqueeze(0).transpose(1, 2))
                        sample2 = sample_from_discretized_mix_logistic(logits2.unsqueeze(0).transpose(1, 2))
                        sample3 = sample_from_discretized_mix_logistic(logits3.unsqueeze(0).transpose(1, 2))
                        sample = torch.cat([sample0,sample1,sample2,sample3],dim=1)
                        # x = torch.FloatTensor([[sample]]).cuda()
                        x = sample.transpose(0, 1)

                    elif self.mode == 'RAW':
                        posterior0 = F.softmax(logits0, dim=1)
                        posterior1 = F.softmax(logits1, dim=1)
                        posterior2 = F.softmax(logits2, dim=1)
                        posterior3 = F.softmax(logits3, dim=1)

                        distrib0 = torch.distributions.Categorical(posterior0)
                        distrib1 = torch.distributions.Categorical(posterior1)
                        distrib2 = torch.distributions.Categorical(posterior2)
                        distrib3 = torch.distributions.Categorical(posterior3)

                        # label -> float
                        sample0 = 2 * distrib0.sample().float() / (self.n_classes - 1.) - 1.
                        sample1 = 2 * distrib1.sample().float() / (self.n_classes - 1.) - 1.
                        sample2 = 2 * distrib2.sample().float() / (self.n_classes - 1.) - 1.
                        sample3 = 2 * distrib3.sample().float() / (self.n_classes - 1.) - 1.
                        sample = torch.cat([sample0.unsqueeze(-1), sample1.unsqueeze(-1), sample2.unsqueeze(-1), sample3.unsqueeze(-1)], dim=-1)
                        # print("sample.shape",sample.shape,"sample0.shape",sample0.shape)
                        output.append(sample)    # final output: (6050,)
                        x = sample

                    else:
                        raise RuntimeError("Unknown model mode value - ", self.mode)

                    if i % 100 == 0: self.gen_display(i, seq_len, b_size, start)

                output = torch.stack(output).transpose(0, 1).transpose(1, 2)
                output = output.cpu().numpy()
                output = output.astype(np.float64)
                # print("output",output)

                if mu_law:
                    output = decode_mu_law(output, self.n_classes, False)

            #########################  MultiBand-WaveRNN   #########################

        if batched:
            output = self.xfade_and_unfold(output, target, overlap)
        else:
            output = output[0]

        np.save(save_path2, output, allow_pickle=False)
        output = mypqmf.synthesis(
            torch.tensor(output, dtype=torch.float).unsqueeze(0)).numpy()  # (batch, sub_band, T//sub_band) -> (batch, 1, T)
        output = output.squeeze()

        # Fade-out at the end to avoid signal cutting out suddenly
        fade_out = np.linspace(1, 0, 20 * self.hop_length)
        output = output[:wave_len]
        output[-20 * self.hop_length:] *= fade_out

        save_wav(output, save_path)

        self.train()

        return output


    def gen_display(self, i, seq_len, b_size, start):
        gen_rate = (i + 1) / (time.time() - start) * b_size / 1000
        pbar = progbar(i, seq_len)
        msg = f'| {pbar} {i*b_size}/{seq_len*b_size} | Batch Size: {b_size} | Gen Rate: {gen_rate:.1f}kHz | '
        stream(msg)

    def get_gru_cell(self, gru):
        gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
        gru_cell.weight_hh.data = gru.weight_hh_l0.data
        gru_cell.weight_ih.data = gru.weight_ih_l0.data
        gru_cell.bias_hh.data = gru.bias_hh_l0.data
        gru_cell.bias_ih.data = gru.bias_ih_l0.data
        return gru_cell

    def pad_tensor(self, x, pad,pad_val,side='both'):  # 引入pad_val-Begee
        # NB - this is just a quick method i need right now
        # i.e., it won't generalise to other shapes/dims
        b, t, c = x.size()
        total = t + 2 * pad if side == 'both' else t + pad
        padded = torch.zeros(b, total, c, device=x.device).fill_(pad_val)
        if side == 'before' or side == 'both':
            padded[:, pad:pad + t, :] = x
        elif side == 'after':
            padded[:, :t, :] = x
        return padded

    def fold_with_overlap(self, x, target, overlap,pad_val):

        ''' Fold the tensor with overlap for quick batched inference.
            Overlap will be used for crossfading in xfade_and_unfold()

        Args:
            x (tensor)    : Upsampled conditioning features.
                            shape=(1, timesteps, features)
            target (int)  : Target timesteps for each index of batch
            overlap (int) : Timesteps for both xfade and rnn warmup

        Return:
            (tensor) : shape=(num_folds, target + 2 * overlap, features)

        Details:
            x = [[h1, h2, ... hn]]

            Where each h is a vector of conditioning features

            Eg: target=2, overlap=1 with x.size(1)=10

            folded = [[h1, h2, h3, h4],
                      [h4, h5, h6, h7],
                      [h7, h8, h9, h10]]
        '''

        _, total_len, features = x.size()

        # Calculate variables needed
        num_folds = (total_len - overlap) // (target + overlap)
        extended_len = num_folds * (overlap + target) + overlap
        remaining = total_len - extended_len

        # Pad if some time steps poking out
        if remaining != 0:
            num_folds += 1
            padding = target + 2 * overlap - remaining
            x = self.pad_tensor(x, padding, pad_val, side='after')

        folded = torch.zeros(num_folds, target + 2 * overlap, features, device=x.device)

        # Get the values for the folded tensor
        for i in range(num_folds):
            start = i * (target + overlap)
            end = start + target + 2 * overlap
            folded[i] = x[:, start:end, :]

        return folded

    def xfade_and_unfold(self, y, target, overlap):

        ''' Applies a crossfade and unfolds into a 1d array.

        Args:
            y (ndarry)    : Batched sequences of audio samples
                            shape=(num_folds, target + 2 * overlap)
                            dtype=np.float64
            overlap (int) : Timesteps for both xfade and rnn warmup

        Return:
            (ndarry) : audio samples in a 1d array
                       shape=(total_len)
                       dtype=np.float64

        Details:
            y = [[seq1],
                 [seq2],
                 [seq3]]

            Apply a gain envelope at both ends of the sequences

            y = [[seq1_in, seq1_target, seq1_out],
                 [seq2_in, seq2_target, seq2_out],
                 [seq3_in, seq3_target, seq3_out]]

            Stagger and add up the groups of samples:

            [seq1_in, seq1_target, (seq1_out + seq2_in), seq2_target, ...]

        '''

        num_folds, length = y.shape
        target = length - 2 * overlap
        total_len = num_folds * (target + overlap) + overlap

        # Need some silence for the rnn warmup
        silence_len = overlap // 2
        fade_len = overlap - silence_len
        silence = np.zeros((silence_len), dtype=np.float64)
        linear = np.ones((silence_len), dtype=np.float64)

        # Equal power crossfade
        t = np.linspace(-1, 1, fade_len, dtype=np.float64)
        fade_in = np.sqrt(0.5 * (1 + t))
        fade_out = np.sqrt(0.5 * (1 - t))

        # Concat the silence to the fades
        fade_in = np.concatenate([silence, fade_in])
        fade_out = np.concatenate([linear, fade_out])

        # Apply the gain to the overlap samples
        y[:, :overlap] *= fade_in
        y[:, -overlap:] *= fade_out

        unfolded = np.zeros((total_len), dtype=np.float64)

        # Loop to add up all the samples
        for i in range(num_folds):
            start = i * (target + overlap)
            end = start + target + 2 * overlap
            unfolded[start:end] += y[i]

        return unfolded

    def get_step(self):
        return self.step.data.item()

    def log(self, path, msg):
        with open(path, 'a') as f:
            print(msg, file=f)

    def load(self, path: Union[str, Path]):
        # Use device of model params as location for loaded state
        device = next(self.parameters()).device
        self.load_state_dict(torch.load(path, map_location=device), strict=False)

    def save(self, path: Union[str, Path]):
        # No optimizer argument because saving a model should not include data
        # only relevant in the training process - it should only be properties
        # of the model itself. Let caller take care of saving optimzier state.
        torch.save(self.state_dict(), path)

    def num_params(self, print_out=True):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        if print_out:
            print('Trainable Parameters: %.3fM' % parameters)
        return parameters

    def _flatten_parameters(self):
        """Calls `flatten_parameters` on all the rnns used by the WaveRNN. Used
        to improve efficiency and avoid PyTorch yelling at us."""
        [m.flatten_parameters() for m in self._to_flatten]
