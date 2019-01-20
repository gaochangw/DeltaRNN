import os
import collections
import modules.models as models
import time
import math
import torch as t
import torch.nn.functional as F
import torch.utils.data.dataloader
import numpy as np
from modules import metric, log
from modules.ctc_dataloader import CTCDataLoader
from tqdm import tqdm
import mkl_random as rnd


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
         phn=48,
         cla_type='GRU',
         cla_layers=2,
         cla_size=512,
         bidirectional=True,
         decoder='beam',
         beam_width=10,
         bestmodel='per',
         cuda=1):

    print("####################################################################")
    print("# Step 3: ASR Testing                                              #")
    print("####################################################################")

   # Set seeds
    rnd.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    print("Initial Seed: ", torch.initial_seed())


    # Create savers, loggers
    logfile = filename + '_' + bestmodel + '_test_asr.csv'  # .csv logfile
    log_dict = collections.OrderedDict()

    # Create iterators
    iterator = CTCDataLoader(trainfile, devfile, testfile)

    # Instantiate Model
    net = models.Model(inp_size=40,
                       cla_type=cla_type,
                       cla_size=cla_size,
                       cla_layers=cla_layers,
                       bidirectional=bidirectional,
                       num_classes=phn+1,
                       cuda=cuda)

    if cuda:
        net = net.cuda()

    # Load trained ASR model
    net.load_state_dict(torch.load('./save/' + filename + '_asr_best_' + bestmodel + '.pt'))
    test_meter = metric.meter(blank=0, source_phn=phn)

    # Epoch loop
    print("Starting testing...")

    # Timer
    start_time = time.time()

    ###########################################################################################################
    # Testing - Iterate batches
    ###########################################################################################################
    # Batch loop - validation
    test_batch = 0
    for bX, b_lenX, bY, b_lenY in tqdm(iterator.iterate(epoch=0,
                                                        h5=testfile,
                                                        cache_size=cache_size,
                                                        batch_size=batch_size,
                                                        mode=iter_mode,
                                                        shuffle_type='none',
                                                        normalization=0,
                                                        enable_gauss=0)):

        # Convert numpy arrays to tensors
        sensor_list = t.from_numpy(bX).cuda()
        sensor_list = sensor_list.transpose(0, 1)

        # Test step
        net = net.eval()

        output = net(sensor_list)
        output = output.transpose(0, 1)  # output.size() = (seq_len x n_batch x n_feature)
        output = F.log_softmax(output, dim=2)

        # Get prediction and best path decoding
        test_meter.extend_guessed_labels(output.cpu().data.numpy(), decoder=decoder, beam_width=beam_width, source_phn=phn)
        test_meter.extend_target_labels(bY=bY, b_lenY=b_lenY)

        acc_PER = test_meter.get_metrics_preserve()
        print('Time: %s | Batch: %3d | PER so far: %.3f%% ' % (timeSince(start_time), test_batch, acc_PER * 100))
        log_dict['epoch'] = 0
        log_dict['stage'] = 'test'
        log_dict['batch'] = test_batch
        log_dict['test_PER'] = acc_PER * 100
        log_dict['test_time'] = timeSince(start_time)
        log.write_log(logfile, log_dict)

        test_batch = test_batch + 1

    test_meter.clear()
    print("Testing Completed...")
    print(" ")
