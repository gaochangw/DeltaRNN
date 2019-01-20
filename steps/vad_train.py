import os
import collections
import modules.models as models
import time
import math
import torch as t
import torch.nn as nn
from torch.nn.functional import log_softmax, softmax
import torch.optim as optim
import torch.utils.data.dataloader
from modules.util import quantizeTensor
import numpy as np
from modules import log
from modules.dataloader import DataLoader
from tqdm import tqdm
import random as rnd
import mkl_random as mkl_rnd


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def main(trainfile='./feat/train_vad.h5',
         testfile='./feat/test_vad.h5',
         swap=0,
         filename='log',
         seed=2,
         batch_size=32,
         nfilt=40,
         window_size=5,
         num_epochs=100,
         opt='ADAM',
         cuda=1,
         lr=3e-4,
         quantize=0,
         m=1,
         n=15):

    print("####################################################################")
    print("# VAD Step 2: Training                                             #")
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
    logfile = filename + '_train_vad.csv'  # .csv logfile

    try:
        os.makedirs(savedir)
    except:
        pass
    log_dict = collections.OrderedDict()

    # Create iterators
    iterator = DataLoader(trainfile)

    # Instantiate Model
    net = models.VADModel(n_feature=nfilt*window_size, num_classes=2)

    if cuda:
        net = net.cuda()

    if opt == 'ADAM':
        optimizer = optim.Adam(net.parameters(), amsgrad=True, lr=lr)
    elif opt == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    criterion = nn.CrossEntropyLoss().cuda()
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
        if 'fc' in name:
            if 'weight' in name:
                #nn.init.kaiming_uniform_(param)
                nn.init.xavier_normal_(param)
                #nn.init.orthogonal_(param)
        if 'bias' in name:  # all biases
            nn.init.constant_(param, 0)

    # Epoch loop
    print("Starting training...")

    # Swap training and test set
    trainfile_swap = trainfile
    testfile_swap = testfile

    if swap:
        trainfile_swap = testfile
        testfile_swap = trainfile

    # Timer
    start_time = time.time()
    for epoch in range(num_epochs):

        train_loss = 0
        train_batches = 0

        ###########################################################################################################
        # Training - Iterate batches
        ###########################################################################################################
        for bX, bY in tqdm(iterator.iterate(epoch=epoch,
                                            h5=trainfile_swap,
                                            batch_size=batch_size,
                                            shuffle_type='random',
                                            normalization=0,
                                            enable_gauss=0)):

            # Convert numpy arrays to tensors
            feature = t.from_numpy(bX).cuda()
            label = t.from_numpy(bY).cuda()

            # Optimization
            optimizer.zero_grad()

            net = net.train()
            output = net(feature)

            loss = criterion(output, label)

            # Get gradients
            loss.backward()

            # Gradient clipping
            #torch.nn.utils.clip_grad_norm_(net.parameters(), 400)

            # Update parameters
            optimizer.step()

            # Quantize weights
            if quantize:
                for name, param in net.named_parameters():
                    if 'weight' in name:
                        param.data = quantizeTensor(param.data, m, n)

            # Increment monitoring variables
            batch_loss = loss.detach()
            train_loss += batch_loss  # Accumulate loss
            train_batches += 1  # Accumulate count so we can calculate mean later

            # Garbage collection to free VRAM
            del feature, loss, output, label

        ###########################################################################################################
        # Testing - Iterate batches
        ###########################################################################################################
        val_loss = 0
        val_batches = 0
        all_output = []
        all_label = []
        # Batch loop - validation
        for bX, bY in tqdm(iterator.iterate(epoch=epoch,
                                            h5=testfile_swap,
                                            batch_size=batch_size,
                                            shuffle_type='none',
                                            normalization=0,
                                            enable_gauss=0)):
            # Convert numpy arrays to tensors
            feature = t.from_numpy(bX).cuda()
            label = t.from_numpy(bY).cuda()

            # Test step
            net = net.eval()

            output = net(feature)
            log_prob = log_softmax(output, dim=1)
            _, result = t.topk(log_prob, 1, dim=1)

            all_output.append(result.detach().cpu().numpy())
            all_label.append(label.detach().cpu().numpy())

            loss = criterion(output, label)

            # Increment monitoring variables
            batch_loss = loss.detach()
            val_loss += batch_loss
            val_batches += 1  # Accumulate count so we can calculate mean later

            # Delete variables
            del loss, output, label

        all_output = np.vstack(all_output)
        all_label = np.hstack(all_label)

        all_output = all_output
        all_label = all_label

        unique, counts = np.unique(all_label, return_counts=True)
        n_sample = all_label.shape[0]
        n_impostor = counts[0]
        n_client = counts[1]

        n_fa = np.sum(all_output[np.where(all_label == 0)])
        n_fr = n_client - np.sum(all_output[np.where(all_label == 1)])

        FAR = n_fa / n_impostor * 100
        FRR = n_fr / n_client * 100

        HTER = (FAR + FRR) / 2

        ACC = (1 - np.sum(np.abs(np.asarray(all_output, dtype=int).reshape(-1) - all_label))/n_sample) * 100

        if epoch == 0:
            val_loss_min = val_loss
        if val_loss <= val_loss_min:
            val_loss_min = val_loss
            torch.save(net.state_dict(), os.path.join(savedir, filename + '_best_vloss_vad.pt'))
            print('>>> saving best V-Loss model from epoch {}'.format(epoch))

        # Get current learning rate
        for param_group in optimizer.param_groups:
            lr_curr = param_group['lr']

        print('Epoch: %3d of %3d | Time: %s | T-Loss: %2.4f | V-Loss: %2.4f | FAR %2.3f | FRR %2.3f | HTER %2.3f | ACC %2.3f |' % (epoch,
                                                                                                                       num_epochs-1,
                                                                                                                       timeSince(start_time),
                                                                                                                       train_loss.item() / train_batches,
                                                                                                                       val_loss.item() / val_batches,
                                                                                                                       FAR,
                                                                                                                       FRR,
                                                                                                                       HTER,
                                                                                                                       ACC))

        log_dict['epoch'] = epoch
        log_dict['#Parameters'] = params
        log_dict['exp_time'] = timeSince(start_time)
        log_dict['train_loss'] = train_loss.item() / train_batches
        log_dict['val_loss'] = val_loss.item() / val_batches
        log_dict['FAR'] = FAR
        log_dict['FRR'] = FRR
        log_dict['HTER'] = HTER
        log_dict['ACC'] = ACC
        log.write_log(logfile, log_dict)

    print("Training Completed...                                               ")
    print(" ")
