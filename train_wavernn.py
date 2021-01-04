import time
import numpy as np
import torch
import math
from torch import optim
import torch.nn.functional as F
from utils.display import stream, simple_table
from utils.dataset import get_vocoder_datasets
from utils.distribution import discretized_mix_logistic_loss
from utils import hparams as hp
from models.fatchord_version import WaveRNN
from gen_wavernn import gen_testset
from utils.paths import Paths
import argparse
from utils import data_parallel_workaround
from utils.checkpoints import save_checkpoint, restore_checkpoint
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  #指定第一块gpu

def main():

    # Parse Arguments
    parser = argparse.ArgumentParser(description='Train WaveRNN Vocoder')
    parser.add_argument('--init_lr', '-il', type=float,  help='[float] override hparams.py learning rate')
    parser.add_argument('--final_lr', '-fl', type=float,  help='[float] override hparams.py learning rate')
    parser.add_argument('--batch_size', '-b', type=int, help='[int] override hparams.py batch size')
    parser.add_argument('--force_train', '-f', action='store_true', help='Forces the model to train past total steps')
    parser.add_argument('--gta', '-g', action='store_true', help='train wavernn on GTA features')
    parser.add_argument('--force_cpu', '-c', action='store_true', help='Forces CPU-only training, even when in CUDA capable environment')
    parser.add_argument('--hp_file', metavar='FILE', default='hparams.py', help='The file to use for the hyperparameters')
    args = parser.parse_args()

    hp.configure(args.hp_file)  # load hparams from file
    if args.batch_size is None:
        args.batch_size = hp.voc_batch_size
    if args.init_lr is None:
        args.init_lr = hp.voc_init_lr
    if args.final_lr is None:
        args.final_lr = hp.voc_final_lr

    paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)

    batch_size = args.batch_size
    force_train = args.force_train
    train_gta = args.gta
    init_lr = args.init_lr
    final_lr = args.final_lr

    if not args.force_cpu and torch.cuda.is_available():
        device = torch.device('cuda')
        if batch_size % torch.cuda.device_count() != 0:
            raise ValueError('`batch_size` must be evenly divisible by n_gpus!')
    else:
        device = torch.device('cpu')
    print('Using device:', device)

    print('\nInitialising Model...\n')

    # Instantiate WaveRNN Model
    voc_model = WaveRNN(rnn_dims=hp.voc_rnn_dims,
                        fc_dims=hp.voc_fc_dims,
                        bits=hp.bits,
                        pad=hp.voc_pad,
                        upsample_factors=hp.voc_upsample_factors,
                        feat_dims=hp.num_mels,
                        compute_dims=hp.voc_compute_dims,
                        res_out_dims=hp.voc_res_out_dims,
                        res_blocks=hp.voc_res_blocks,
                        hop_length=hp.hop_length,
                        sample_rate=hp.sample_rate,
                        pad_val=hp.voc_pad_val,  # 引入pad_val-Begee
                        mode=hp.voc_mode).to(device)

    # Check to make sure the hop length is correctly factorised
    #########################  MultiBand-WaveRNN   #########################
    # assert np.cumprod(hp.voc_upsample_factors)[-1] == hp.hop_length // 4
    #########################  MultiBand-WaveRNN   #########################

    optimizer = optim.Adam(voc_model.parameters())
    restore_checkpoint('voc', paths, voc_model, optimizer, create_if_missing=True)

    train_set, test_set = get_vocoder_datasets(paths.data, batch_size, train_gta)

    total_steps = 10_000_000 if force_train else hp.voc_total_steps

    simple_table([('Remaining', str((total_steps - voc_model.get_step())//1000) + 'k Steps'),
                  ('Batch Size', batch_size),
                  ('Init_lr', hp.voc_init_lr),
                  ('Sequence Len', hp.voc_seq_len),
                  ('GTA Train', train_gta)])

    loss_func = F.cross_entropy if voc_model.mode == 'RAW' else discretized_mix_logistic_loss

    voc_train_loop(paths, voc_model, loss_func, optimizer, train_set, test_set, init_lr, final_lr,total_steps) # 初始学习率与最终学习率-Begee

    print('Training Complete.')
    print('To continue training increase voc_total_steps in hparams.py or use --force_train')

def cosine_decay(init_val, final_val, step, decay_steps):
    alpha = final_val / init_val
    cosine_decay = 0.5 * (1 + math.cos(math.pi * step / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha
    return init_val * decayed


def adjust_learning_rate(optimizer, epoch, epochs, init_lr, final_lr):

    lr = cosine_decay(init_lr, final_lr, epoch, epochs)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def voc_train_loop(paths: Paths, model: WaveRNN, loss_func, optimizer, train_set, test_set, init_lr, final_lr, total_steps):
    # Use same device as model parameters
    device = next(model.parameters()).device

    # for g in optimizer.param_groups: g['lr'] = lr

    total_iters = len(train_set)
    epochs = (total_steps - model.get_step()) // total_iters + 1

    for e in range(1, epochs + 1):

        adjust_learning_rate(optimizer, e, epochs, init_lr, final_lr)  # 初始学习率与最终学习率-Begee
        start = time.time()
        running_loss = 0.

        for i, (x, y, m) in enumerate(train_set, 1):
            x, m, y = x.to(device), m.to(device), y.to(device)  # x/y: (Batch, sub_bands, T)

#########################  MultiBand-WaveRNN   #########################
            if hp.voc_multiband:
                y0 = y[:, 0, :].squeeze(0).unsqueeze(-1)  # y0/y1/y2/y3: (Batch, T, 1)
                y1 = y[:, 1, :].squeeze(0).unsqueeze(-1)
                y2 = y[:, 2, :].squeeze(0).unsqueeze(-1)
                y3 = y[:, 3, :].squeeze(0).unsqueeze(-1)

                y_hat = model(x, m)  # (Batch, T, num_classes, sub_bands)

                if model.mode == 'RAW':
                    y_hat0 = y_hat[:, :, :, 0].transpose(1,2).unsqueeze(-1)  # (Batch, num_classes, T, 1)
                    y_hat1 = y_hat[:, :, :, 1].transpose(1,2).unsqueeze(-1)
                    y_hat2 = y_hat[:, :, :, 2].transpose(1,2).unsqueeze(-1)
                    y_hat3 = y_hat[:, :, :, 3].transpose(1,2).unsqueeze(-1)

                elif model.mode == 'MOL':
                    y0 = y0.float()
                    y1 = y1.float()
                    y2 = y2.float()
                    y3 = y3.float()

                loss = loss_func(y_hat0, y0) + loss_func(y_hat1, y1) + loss_func(y_hat2, y2) + loss_func(y_hat3, y3)

            #########################  MultiBand-WaveRNN   #########################


            optimizer.zero_grad()
            loss.backward()

            if hp.voc_clip_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hp.voc_clip_grad_norm)
                if np.isnan(grad_norm):
                    print('grad_norm was NaN!')
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / i

            speed = i / (time.time() - start)

            step = model.get_step()
            k = step // 1000

            if step % hp.voc_checkpoint_every == 0:
                gen_testset(model, test_set, hp.voc_gen_at_checkpoint, hp.voc_gen_batched,
                            hp.voc_target, hp.voc_overlap, paths.voc_output)
                ckpt_name = f'wave_step{k}K'
                save_checkpoint('voc', paths, model, optimizer,
                                name=ckpt_name, is_silent=True)

            msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Loss: {avg_loss:.4f} | {speed:.1f} steps/s | Step: {k}k | '
            stream(msg)

        # Must save latest optimizer state to ensure that resuming training
        # doesn't produce artifacts
        save_checkpoint('voc', paths, model, optimizer, is_silent=True)
        model.log(paths.voc_log, msg)
        print(' ')


if __name__ == "__main__":
    main()
