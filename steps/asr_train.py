import os
import collections
import modules.models as models
import time
import math
import torch as t
from warpctc_pytorch import CTCLoss
from modules.util import quantizeTensor
#from torch.nn import CTCLoss
import torch.nn as nn
from torch.nn.functional import log_softmax, softmax
import torch.optim as optim
import torch.utils.data.dataloader
import numpy as np
from modules import metric, log
from modules.ctc_dataloader import CTCDataLoader
from tqdm import tqdm
import random as rnd
import mkl_random as mkl_rnd
import pandas as pd


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def main(trainfile='./feat/train.h5',
         devfile='./feat/dev.h5',
         testfile='./feat/test.h5',
         filename='debug',
         seed=2,
         cache_size=10000,
         batch_size=32,
         nfilt=40,
         iter_mode='batch',
         num_epochs=100,
         phn=48,
         cla_type='GRU',
         cla_layers=2,
         cla_size=512,
         bidirectional=1,
         opt='adam',
         decay_rate=0.8,
         decay_epoch=10,
         cuda=1,
         lr=3e-4,
         val=0,
         quantize=0,
         m=1,
         n=15):

    def learning_rate_decay(optimizer, init_learning_rate, factor, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
        lr = init_learning_rate * (factor ** (epoch // decay_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    print("####################################################################")
    print("# ASR Step 2: Training                                             #")
    print("####################################################################")

   # Set seeds
    rnd.seed(seed)
    mkl_rnd.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    print("Initial Seed: ", torch.initial_seed())


    # Create savers, loggers
    savedir = 'save'  # save model directory
    logfile = filename + '_train_asr.csv'  # .csv logfile

    try:
        os.makedirs(savedir)
    except:
        pass
    log_dict = collections.OrderedDict()

    # Create iterators
    iterator = CTCDataLoader(trainfile, devfile, testfile)

    # Instantiate Model
    net = models.Model(inp_size=nfilt,
                       cla_type=cla_type,
                       cla_size=cla_size,
                       cla_layers=cla_layers,
                       bidirectional=bidirectional,
                       num_classes=phn+1,
                       cuda=cuda)

    if cuda:
        net = net.cuda()

    if opt == 'ADAM':
        optimizer = optim.Adam(net.parameters(), amsgrad=True, lr=lr)
    elif opt == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    #criterion = CTCLoss(reduction='sum')
    criterion = CTCLoss()
    print(net)

    # Print parameter count
    params = 0
    for param in list(net.parameters()):
        sizes = 1
        for el in param.size():
            sizes = sizes * el
        params += sizes
    print('::: # network parameters: ' + str(params))

    # Weight initialization --> accelerate training with Xavier
    for name, param in net.named_parameters():
        if 'rnn' in name:
            if 'weight' in name:
                nn.init.orthogonal_(param)
        if 'fc' in name:
            if 'weight' in name:
                nn.init.orthogonal_(param)
        if 'bias' in name:  # all biases
            nn.init.constant_(param, 0)
        if cla_type == 'LSTM':  # only LSTM biases
            if ('bias_ih' in name) or ('bias_hh' in name):
                no4 = int(len(param) / 4)
                no2 = int(len(param) / 2)
                nn.init.constant_(param, 0)
                nn.init.constant_(param[no4:no2], 1)

    test_meter = metric.meter(blank=0, source_phn=phn)
    val_meter = metric.meter(blank=0, source_phn=phn)

    # Epoch loop
    print("Starting training...")

    # Create PHN-48 to PHN-39 mapping dict
    phn_map_48_39 = pd.read_csv('./data/phn_map_48_39.csv')
    dict_48_39_int = {}
    for idx, x in enumerate(phn_map_48_39['phn-48-int']):
        dict_48_39_int[x + 1] = int(
            phn_map_48_39['phn-39-int'][idx]) + 1  # Create PHN-48 to PHN-39 conversion dict

    columns = ['loss', 'output_sum', 'rnn_in', 'rnn_out']
    loss_list = []

    # Timer
    start_time = time.time()
    for epoch in range(num_epochs):

        learning_rate_decay(optimizer, lr, decay_rate, epoch)
        train_loss = 0
        train_batches = 0

        ###########################################################################################################
        # Training - Iterate batches
        ###########################################################################################################
        for bX, b_lenX, bY, b_lenY in tqdm(iterator.iterate(epoch=epoch,
                                                            h5=trainfile,
                                                            cache_size=cache_size,
                                                            batch_size=batch_size,
                                                            mode=iter_mode,
                                                            shuffle_type='random',
                                                            normalization=0,
                                                            enable_gauss=0)):
            #if train_batches == 0:
            #    print(bX.shape)

            # Convert numpy arrays to tensors
            sensor_list = t.from_numpy(bX).cuda()
            sensor_list = sensor_list.transpose(0, 1)
            feature_length = t.from_numpy(b_lenX)
            label = t.from_numpy(bY)
            label_length = t.from_numpy(b_lenY)

            # Optimization
            optimizer.zero_grad()

            net = net.train()

            output = net(sensor_list)
            output = output.transpose(0, 1)  # output.size() = (seq_len x n_batch x n_feature)


            #log_probs = log_softmax(output, dim=2)
            # print("Output sizes: ", output.size())

            loss = criterion(output, label, feature_length, label_length)
            # row = {}
            # row['loss'] = str(loss.item())
            # row['output_sum'] = str(output.sum().item())
            # row['rnn_in'] = str(rnn_in)
            # row['rnn_out'] = str(rnn_out)
            # loss_list.append(row)

            # Get gradients
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(net.parameters(), 400)

            # Update parameters
            optimizer.step()

            # Quantize weights
            if quantize:
                for name, param in net.named_parameters():
                    if 'rnn' in name:
                        if 'weight' in name:
                            param.data = quantizeTensor(param.data, m, n)

            # Increment monitoring variables
            batch_loss = loss.detach()
            train_loss += batch_loss  # Accumulate loss
            train_batches += 1  # Accumulate count so we can calculate mean later

            # Garbage collection to free VRAM
            del sensor_list, loss, output, feature_length, label, label_length

        # df = pd.DataFrame(loss_list, columns=columns)
        # description_file = os.path.join('./', 'description.csv')
        # df.to_csv(description_file, index=False)

        ###########################################################################################################
        # Validate - Iterate batches
        ###########################################################################################################
        val_loss = 0
        val_batches = 0
        if val:
            # Batch loop - validation
            for bX, b_lenX, bY, b_lenY in tqdm(iterator.iterate(epoch=epoch,
                                                                h5=devfile,
                                                                cache_size=cache_size,
                                                                batch_size=batch_size,
                                                                mode=iter_mode,
                                                                shuffle_type='none',
                                                                normalization=0,
                                                                enable_gauss=0)):

                # Convert numpy arrays to tensors
                sensor_list = t.from_numpy(bX).cuda()
                feature_length = t.from_numpy(b_lenX)
                label = t.from_numpy(bY)
                label_length = t.from_numpy(b_lenY)
                sensor_list = sensor_list.transpose(0, 1)

                # Test step
                net = net.eval()

                output = net(sensor_list)
                output = output.transpose(0, 1)  # output.size() = (seq_len x n_batch x n_feature)

                loss = criterion(output, label, feature_length, label_length)

                # Get prediction and best path decoding
                val_meter.extend_guessed_labels(output.cpu().data.numpy())
                val_meter.extend_target_labels(bY=bY, b_lenY=b_lenY)

                # Increment monitoring variables
                batch_loss = loss.detach()
                val_loss += batch_loss
                val_batches += 1  # Accumulate count so we can calculate mean later

                # Delete variables
                del sensor_list, loss, output, feature_length, label, label_length

            val_PER = val_meter.get_metrics()

        if epoch == 0:
            val_loss_min = val_loss
        if val_loss <= val_loss_min:
            val_loss_min = val_loss
            torch.save(net.state_dict(), os.path.join(savedir, filename + '_asr_best_vloss.pt'))
            print('>>> saving best V-Loss model from epoch {}'.format(epoch))

        # if epoch == 0:
        #     val_PER_min = val_PER
        # if val_PER <= val_PER_min:
        #     val_PER_min = val_PER
        #     torch.save(net.state_dict(), os.path.join(savedir, filename + '_asr_best_vper.pt'))
        #     print('>>> saving best V-PER model from epoch {}'.format(epoch))



        ###########################################################################################################
        # Testing - Iterate batches
        ###########################################################################################################
        if (epoch + 1) % 1 == 0:
            # Batch loop - validation
            for bX, b_lenX, bY, b_lenY in tqdm(iterator.iterate(epoch=epoch,
                                                                h5=testfile,
                                                                cache_size=cache_size,
                                                                batch_size=batch_size,
                                                                mode=iter_mode,
                                                                shuffle_type='random',
                                                                normalization=0,
                                                                enable_gauss=0)):

                # Convert numpy arrays to tensors
                sensor_list = t.from_numpy(bX).cuda()
                sensor_list = sensor_list.transpose(0, 1)

                # Test step
                net = net.eval()

                output = net(sensor_list)
                output = output.transpose(0, 1)  # output.size() = (seq_len x n_batch x n_feature)

                # Get prediction and best path decoding
                test_meter.extend_guessed_labels(output.cpu().data.numpy())
                test_meter.extend_target_labels(bY=bY, b_lenY=b_lenY)

                # Delete variables
                del sensor_list, output

            test_PER = test_meter.get_metrics()

            # Get current learning rate
            for param_group in optimizer.param_groups:
                lr_curr = param_group['lr']

            if val:
                print('Epoch: %3d of %3d | Time: %s | LR: %1.8f | T-Loss: %5.2f | V-Loss: %5.2f | PER: %.3f |' % (epoch,
                                                                                                                  num_epochs-1,
                                                                                                                  timeSince(start_time),
                                                                                                                  lr_curr,
                                                                                                                  train_loss / train_batches,
                                                                                                                  val_loss / val_batches,
                                                                                                                  test_PER * 100))
            else:
                print('Epoch: %3d of %3d | Time: %s | LR: %1.8f | T-Loss: %5.2f | PER: %.3f |' % (epoch,
                                                                                                  num_epochs-1,
                                                                                                  timeSince(start_time),
                                                                                                  lr_curr,
                                                                                                  train_loss / train_batches,
                                                                                                  test_PER * 100))

            log_dict['epoch'] = epoch
            log_dict['stage'] = 'train'
            log_dict['quantize'] = quantize
            if quantize:
                log_dict['m'] = m
                log_dict['n'] = n
            log_dict['train_loss'] = train_loss.item() / train_batches
            if val:
                log_dict['val_loss'] = val_loss.item() / val_batches
            log_dict['test_PER'] = test_PER * 100
            log_dict['#Parameters'] = params
            log_dict['exp_time'] = timeSince(start_time)
            log.write_log(logfile, log_dict)

        if epoch == 0:
            test_PER_min = test_PER
        if test_PER <= test_PER_min:
            test_PER_min = test_PER
            torch.save(net.state_dict(), os.path.join(savedir, filename + '_asr_best_per.pt'))
            print('>>> saving best PER model from epoch {}'.format(epoch))

    print("Training Completed...                                               ")
    print(" ")
