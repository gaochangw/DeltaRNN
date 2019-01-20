__author__ = "Chang Gao, Manish Prajapat, Frederic Debraine, and Martyna Dziadosz"
__copyright__ = "Copyright 2018 to all authors"
__credits__ = ["Stefan Braun"]
__license__ = "Private"
__version__ = "0.1.0"
__maintainer__ = "Chang Gao, Manish Prajapat, Frederic Debraine, and Martyna Dziadosz"
__email__ = "chang.gao@uzh.ch"
__status__ = "Prototype"

from steps import asr_data_prep, asr_feat_ext, asr_train, asr_test
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a GRU network.')
    parser.add_argument('--path_dataset', default='/DATA/TIMIT/TIMIT', help='training hdf5 file')
    parser.add_argument('--experiment', default='debug', help='Name of experiment for log and model folders')
    parser.add_argument('--filename', default='debug', help='Filename to save model and log to.')
    parser.add_argument('--seed', default=2, type=int, help='Initialize the random seed of the run (for reproducibility).')
    parser.add_argument('--cache_size', default=5000, type=int, help='Max timesteps per batch.')
    parser.add_argument('--batch_size', default=16, type=int, help='Static batch size.')
    parser.add_argument('--iter_mode', default='cache', help='Dynamic batch size.')
    parser.add_argument('--num_epochs', default=3, type=int, help='Number of epochs to train for.')
    parser.add_argument('--frame_size', default=0.025, type=float, help='Number of epochs to train for.')
    parser.add_argument('--frame_stride', default=0.01, type=float, help='Number of epochs to train for.')
    parser.add_argument('--nfilt', default=40, type=int, help='Number of filter banks used as features.')
    parser.add_argument('--phn', default=48, type=int, help='Number of phonemes used in training (61 or 48)')
    parser.add_argument('--cla_type', default='GRU', help='RNN type in all classification module')
    parser.add_argument('--cla_layers', default=5, type=int, help='number of classification layers')
    parser.add_argument('--cla_size', default=512, type=int, help='Size of classification layer')
    parser.add_argument('--bidirectional', default=0, type=int, help='Whether use bidirectional RNNs')
    parser.add_argument('--opt', default='ADAM', help='Which optimizer to use (ADAM or SGD)')
    parser.add_argument('--lr', default=3e-4, type=float, help='Learning rate')
    parser.add_argument('--decay_rate', default=0.8, type=float, help='Learning rate')
    parser.add_argument('--decay_epoch', default=5, type=float, help='Learning rate')
    parser.add_argument('--step', default=0, type=int, help='Which step to start from')
    parser.add_argument('--val', default=1, type=int, help='Whether to do validation')
    parser.add_argument('--decoder', default='beam', help='Whether to do validation')
    parser.add_argument('--beam_width', default=10, type=int, help='Whether to do validation')
    parser.add_argument('--cuda', default=1, type=int, help='Use GPU yes/no')
    parser.add_argument('--run_through', default=1, type=int, help='Whether run through rest steps')
    parser.add_argument('--quantize', default=0, type=int, help='Whether quantize the network weights')
    parser.add_argument('--m', default=1, type=int, help='Number of integer bits before decimal point')
    parser.add_argument('--n', default=7, type=int, help='Number of fraction bits after decimal point')
    parser.add_argument('--bestmodel', default='per', help='Best model used in testing, either "per", or "vloss"')
    args = parser.parse_args()
    print(args.bidirectional)
    path_root = os.path.dirname(os.path.abspath(__file__))

    # Step 0 - TIMIT Data Preparation
    if (args.step <= 0 and args.run_through) or args.step == 0:
        asr_data_prep.main(path_dataset=args.path_dataset,
                             proj_root=path_root)

    # Step 1 - Feature Extraction
    if (args.step <= 1 and args.run_through) or args.step == 1:
        asr_feat_ext.main(root=path_root,
                          nfilt=args.nfilt,
                          frame_size=args.frame_size,
                          frame_stride=args.frame_stride,
                          MFCC=False,
                          phn=args.phn)
    
    # Step 2 - ASR Training
    if (args.step <= 2 and args.run_through) or args.step == 2:
        asr_train.main(trainfile=os.path.join(path_root, 'feat', 'train.h5'),
                       devfile=os.path.join(path_root, 'feat', 'dev.h5'),
                       testfile=os.path.join(path_root, 'feat', 'test.h5'),
                       filename=args.filename,
                       seed=args.seed,
                       cache_size=args.cache_size,
                       batch_size=args.batch_size,
                       nfilt=args.nfilt,
                       iter_mode=args.iter_mode,
                       num_epochs=args.num_epochs,
                       phn=args.phn,
                       cla_type=args.cla_type,
                       cla_layers=args.cla_layers,
                       cla_size=args.cla_size,
                       bidirectional=args.bidirectional,
                       opt=args.opt,
                       decay_rate=args.decay_rate,
                       decay_epoch=args.decay_epoch,
                       cuda=args.cuda,
                       lr=args.lr,
                       val=args.val,
                       quantize=args.quantize,
                       m=args.m,
                       n=args.n)

    # Step 3 - ASR Testing
    if (args.step <= 3 and args.run_through) or args.step == 3:
        asr_test.main(trainfile=os.path.join(path_root, 'feat', 'train.h5'),
                      devfile=os.path.join(path_root, 'feat', 'dev.h5'),
                      testfile=os.path.join(path_root, 'feat', 'test.h5'),
                      filename=args.filename,
                      seed=args.seed,
                      cache_size=args.cache_size,
                      batch_size=args.batch_size,
                      nfilt=args.nfilt,
                      iter_mode=args.iter_mode,
                      phn=args.phn,
                      cla_type=args.cla_type,
                      cla_layers=args.cla_layers,
                      cla_size=args.cla_size,
                      bidirectional=args.bidirectional,
                      decoder=args.decoder,
                      beam_width=args.beam_width,
                      bestmodel=args.bestmodel,
                      cuda=args.cuda)
