__author__ = "Chang Gao, Manish Prajapat, Frederic Debraine, and Martyna Dziadosz"
__copyright__ = "Copyright 2018 to all authors"
__credits__ = ["Stefan Braun"]
__license__ = "Private"
__version__ = "0.1.0"
__maintainer__ = "Chang Gao, Manish Prajapat, Frederic Debraine, and Martyna Dziadosz"
__email__ = "chang.gao@uzh.ch"
__status__ = "Prototype"

from steps import vad_data_prep, vad_feat_ext, vad_train
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a GRU network.')
    parser.add_argument('--path_dataset', default='/DATA/QUT-NOISE', help='training hdf5 file')
    parser.add_argument('--swap', default=0, type=int, help='swap')
    parser.add_argument('--filename', default='debug', help='Filename to save model and log to.')
    parser.add_argument('--seed', default=2, type=int, help='Initialize the random seed of the run (for reproducibility).')
    parser.add_argument('--batch_size', default=512, type=int, help='Static batch size.')
    parser.add_argument('--frame_size', default=0.1, type=float, help='Frame size')
    parser.add_argument('--frame_stride', default=0.1, type=float, help='Frame stride.')
    parser.add_argument('--window_size', default=1, type=int, help='Number of frame concat together as an augmented feature.')
    parser.add_argument('--nfilt', default=40, type=int, help='Number of filter banks.')
    parser.add_argument('--num_epochs', default=10, type=int, help='Number of epochs to train for.')
    parser.add_argument('--opt', default='ADAM', help='Which optimizer to use (ADAM or SGD)')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--step', default=2, type=int, help='Which step to start from')
    parser.add_argument('--cuda', default=1, type=int, help='Use GPU yes/no')
    parser.add_argument('--run_through', default=1, type=int, help='Whether run through rest steps')
    parser.add_argument('--quantize', default=0, type=int, help='Whether quantize the network weights')
    parser.add_argument('--m', default=8, type=int, help='Number of integer bits before decimal point')
    parser.add_argument('--n', default=8, type=int, help='Number of fraction bits after decimal point')
    args = parser.parse_args()

    path_root = os.path.dirname(os.path.abspath(__file__))

    # Step 0 - QUT-NOISE-TIMIT Data Preparation
    if (args.step <= 0 and args.run_through) or args.step == 0:
        vad_data_prep.main(path_dataset=args.path_dataset,
                           proj_root=path_root)

    # Step 1 - QUT-NOISE-TIMIT Feature Extraction
    if (args.step <= 1 and args.run_through) or args.step == 1:
        vad_feat_ext.main(root=path_root,
                          nfilt=args.nfilt,
                          frame_size=args.frame_size,
                          frame_stride=args.frame_stride,
                          window_size=args.window_size,
                          MFCC=False)

    # Step 2 - VAD Training
    if (args.step <= 2 and args.run_through) or args.step == 2:
        vad_train.main(trainfile=os.path.join(path_root, 'feat', 'train_vad.h5'),
                       testfile=os.path.join(path_root, 'feat', 'test_vad.h5'),
                       swap=args.swap,
                       filename=args.filename,
                       seed=args.seed,
                       batch_size=args.batch_size,
                       nfilt=args.nfilt,
                       window_size=args.window_size,
                       num_epochs=args.num_epochs,
                       opt=args.opt,
                       cuda=args.cuda,
                       lr=args.lr,
                       quantize=args.quantize,
                       m=args.m,
                       n=args.n)
