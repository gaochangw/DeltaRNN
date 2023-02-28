import os
import warnings

import numpy as np
import torch
import pandas as pd

from project import Project
from utils import util


def gen_paths(proj):
    model_id, pretrain_model_id = gen_model_id(proj)

    save_dir = os.path.join('save', proj.dataset, proj.step)  # Best model save dir
    log_dir_hist = os.path.join('log', proj.dataset, proj.step, 'hist')  # Log dir to save training history
    log_dir_best = os.path.join('log', proj.dataset, proj.step, 'best')  # Log dir to save info of the best epoch
    log_dir_test = os.path.join('log', proj.dataset, proj.step, 'test')
    dir_paths = (save_dir, log_dir_hist, log_dir_best, log_dir_test)

    # File Paths
    if model_id is not None:
        logfile_hist = os.path.join(log_dir_hist, model_id + '.csv')  # .csv logfile_hist
        logfile_best = os.path.join(log_dir_best, model_id + '.csv')  # .csv logfile_hist
        logfile_test = os.path.join(log_dir_test, model_id + '.csv')  # .csv logfile_hist
        save_file = os.path.join(save_dir, model_id + '.pt')
    if model_id is not None:
        file_paths = (save_file, logfile_hist, logfile_best, logfile_test)
    else:
        file_paths = None

    # Pretrain Model Path
    if pretrain_model_id is not None:
        pretrain_file = os.path.join('./save', proj.dataset, 'pretrain', pretrain_model_id + '.pt')
    else:
        pretrain_file = None

    return dir_paths, file_paths, pretrain_file


class Logger:
    def __init__(self, logfile):
        self.logfile = logfile
        self.list_header = []
        self.loglist = []

    def add_row(self, list_header, list_value):
        self.list_header = list_header
        row = {}
        for header, value in zip(list_header, list_value):
            row[header] = value
        self.loglist.append(row)

    def write_log(self, logfile=None):
        if len(self.list_header) == 0:
            warnings.warn("DataFrame columns not defined. Call add_row for at least once...", RuntimeWarning)
        else:
            df = pd.DataFrame(self.loglist, columns=self.list_header)
            if logfile is not None:
                df.to_csv(logfile, index=False)
            else:
                df.to_csv(self.logfile, index=False)

    def write_log_idx(self, idx, logfile=None):
        if len(self.list_header) == 0:
            warnings.warn("DataFrame columns not defined. Call add_row for at least once...", RuntimeWarning)
        else:
            loglist_best = [self.loglist[idx]]
            df = pd.DataFrame(loglist_best, columns=self.list_header)
            if logfile is not None:
                df.to_csv(logfile, index=False)
            else:
                df.to_csv(self.logfile, index=False)


def gen_trainset_name(proj):
    # Feat name
    dataset_id = '_FT_' + proj.feat_type \
                 + '_NF_' + str(proj.n_filt) \
                 + '_Q_' + str(proj.q_factors) \
                 + '_SI_' + str(proj.frame_size) \
                 + '_ST_' + str(proj.frame_stride) \
                 + '_FL_' + str(proj.freq_low) \
                 + '_FH_' + str(proj.freq_high) \
                 + '_AN_' + str(proj.augment_noise)

    trainfile = 'TRAIN' + dataset_id

    if proj.augment_noise:
        dataset_id += '_SNR_' + str(proj.target_snr)
        devfile = 'DEV' + dataset_id
    else:
        dataset_id += '_SNR_' + '100'
        devfile = 'DEV' + dataset_id

    trainfile += '.h5'
    devfile += '.h5'

    return dataset_id, trainfile, devfile


def gen_testset_name(proj):
    # Feat name
    dataset_id = '_FT_' + proj.feat_type \
                 + '_NF_' + str(proj.n_filt) \
                 + '_Q_' + str(proj.q_factors) \
                 + '_SI_' + str(proj.frame_size) \
                 + '_ST_' + str(proj.frame_stride) \
                 + '_FL_' + str(proj.freq_low) \
                 + '_FH_' + str(proj.freq_high) \
                 + '_AN_' + str(proj.augment_noise)

    if proj.augment_noise:
        dataset_id += '_SNR_' + str(proj.target_snr)
        testfile = 'TEST' + dataset_id
    else:
        dataset_id += '_SNR_' + '100'
        testfile = 'TEST' + dataset_id

    testfile += '.h5'

    return testfile


def gen_model_id(proj):
    # Setting Description
    str_setting = 'S_' + f"{proj.seed:d}"

    # Feature Description
    trainset_name, _, _ = gen_trainset_name(proj)

    str_feat = trainset_name \
               + '_LOG_' + str(proj.log_feat)

    # Architecture Description
    def gen_net_arch(model_name):
        str_net_arch = "_I_" + f"{proj.input_size:d}" \
                       + '_L_' + f"{proj.rnn_layers:d}" \
                       + '_H_' + f"{proj.rnn_size:d}" \
                       + '_M_' + model_name
        # Add FC Layer
        if proj.fc_extra_size > 0:
            str_net_arch += '_FC_' + f"{proj.fc_extra_size:d}"

        # Add Number of Classes
        str_net_arch += '_C_' + f"{proj.num_classes:d}"

        # Quantization of Activation
        if proj.qa:
            str_net_arch += '_QA_' + f"{proj.aqi:d}" + '.' + f"{proj.aqf:d}"
        if proj.qc:
            str_net_arch += '_QC_' + f"{proj.cqi:d}" + '.' + f"{proj.cqf:d}"
        # Quantization of Weights
        if proj.qw:
            str_net_arch += '_QW_' + f"{proj.wqi:d}" + '.' + f"{proj.wqf:d}"
        # Quantization of Classification Layer Weights
        if proj.qcw:
            str_net_arch += '_QCW_' + f"{proj.cwqi:d}" + '.' + f"{proj.cwqf:d}"
        # CBTD
        if proj.cbtd:
            str_net_arch += '_TD_' + f"{proj.gamma_rnn:.2f}"
            str_net_arch += '_AS_' + f"{proj.num_array_pe:d}"
        return str_net_arch

    str_net_arch = gen_net_arch(proj.model_name)

    # Pretrain Model ID
    str_net_arch_pretrain = gen_net_arch(proj.model_pretrain)
    pretrain_model_id = str_setting + str_net_arch_pretrain
    pretrain_model_id = pretrain_model_id.replace("Delta",
                                                  "")  # Remove "Delta" from the model ID to load the non-delta network

    # Delta Network
    if 'delta' in proj.model_name:
        str_net_arch += '_TX_' + f"{proj.thx:.2f}" + '_TH_' + f"{proj.thh:.2f}"

    # Model ID
    # model_id = str_custom + str_setting + str_feat + str_net_arch
    model_id = str_setting + str_net_arch

    return model_id, pretrain_model_id


def write_log(args, logger, tb_writer, model_id, train_stat, val_stat, test_stat, net, optimizer, epoch, time_curr,
              alpha, retrain):
    def get_dict_keyword():
        dict_keyword2label = {}
        dict_label2keyword = {}
        label_list = pd.read_csv('./data/' + args.dataset_name + '/label_list_10_test.csv')
        for row in label_list.itertuples():
            dict_keyword2label[str(row.keyword)] = row.label
            dict_label2keyword[row.label] = {str(row.keyword)}
        return dict_keyword2label, dict_label2keyword

    # Get Dictionaries for Conversion between Keywords & Labels
    dict_keyword2label, dict_label2keyword = get_dict_keyword()

    # Get parameter count
    n_param = 0
    for name, param in net.named_parameters():
        sizes = 1
        for el in param.size():
            sizes = sizes * el
        n_param += sizes

    # Evaluate RNN Weight Sparsity
    n_nonzero_weight_elem = 0
    n_weight_elem = 0
    for name, param in net.named_parameters():
        if 'rnn' in name:
            if 'weight' in name:
                n_nonzero_weight_elem += len(param.data.nonzero())
                n_weight_elem += param.data.nelement()
    sp_w_rnn = 1 - (n_nonzero_weight_elem / n_weight_elem)

    # Evaluate FC Layer Weight Sparsity
    sp_w_fc = 0
    if args.fc_extra_size:
        n_nonzero_weight_elem = 0
        n_weight_elem = 0
        for name, param in net.named_parameters():
            if 'fc_extra' in name:
                if 'weight' in name:
                    n_nonzero_weight_elem += len(param.data.nonzero())
                    n_weight_elem += param.data.nelement()
        sp_w_fc = 1 - (n_nonzero_weight_elem / n_weight_elem)

    # Get current learning rate
    lr_curr = 0
    if optimizer is not None:
        for param_group in optimizer.param_groups:
            lr_curr = param_group['lr']

    # Create Log List
    list_log_headers = ['EPOCH', 'TIME', 'N_PARAM', 'alpha']
    list_log_values = [epoch, time_curr, n_param, alpha]
    if train_stat is not None:
        list_log_headers.append('LR')
        list_log_values.append(lr_curr)
    if train_stat is not None:
        list_log_headers.append('LOSS_TRAIN')
        list_log_values.append(train_stat['loss'])
    if val_stat is not None:
        list_log_headers.append('LOSS_VAL')
        list_log_values.append(val_stat['loss'])
        if args.score_val:
            list_log_headers.extend(['ACC-VAL', 'SP-DX', 'SP-DH'])
            list_log_values.extend([val_stat['f1_score_micro'], val_stat['sp_dx'], val_stat['sp_dh']])
    if test_stat is not None:
        list_log_headers.extend(['LOSS_TEST', 'ACC-TEST', 'SP-DX', 'SP-DH'])
        list_log_values.extend([test_stat['loss'], test_stat['f1_score_micro'], test_stat['sp_dx'], test_stat['sp_dh']])
        list_log_headers.extend([key for key in dict_keyword2label.keys()])
        list_log_values.extend(test_stat['tpr'])

    # Write Log
    logger.add_row(list_log_headers, list_log_values)
    logger.write_csv()

    # Print Info
    if retrain:
        n_epochs = args.n_epochs_retrain
    else:
        n_epochs = args.n_epochs

    str_print = f"Epoch: {epoch + 1:3d} of {n_epochs:3d} | Time: {time_curr:s} | LR: {lr_curr:1.5f} | Sp.W {sp_w_rnn * 100:3.2f}%% | Sp.Wfc {sp_w_fc * 100:3.2f}%% |\n"
    if train_stat is not None:
        str_print += f"    | Train-Loss: {train_stat['loss']:4.2f} | Train-Reg: {train_stat['reg']:4.2f} |\n"
    if val_stat is not None:
        str_print += f"    | Val-Loss: {val_stat['loss']:4.3f}"
        if args.score_val:
            str_print += f" | Val-F1: {val_stat['f1_score_micro'] * 100:3.2f} | Val-Sp-dx: {val_stat['sp_dx'] * 100:3.2f} | Val-Sp" \
                         f"-dh {val_stat['sp_dh'] * 100:3.2f} |"
        str_print += '\n'
    if test_stat is not None:
        str_print += f"    | Test-Loss: {test_stat['loss']:4.3f}"
        if args.score_test:
            str_print += f" | Test-F1: {test_stat['f1_score_micro'] * 100:3.2f} | Test-Sp-dx: {test_stat['sp_dx'] * 100:3.2f} | Test-Sp" \
                         f"-dh: {test_stat['sp_dh'] * 100:3.2f} | "
    print(str_print)

    # Tensorboard
    if tb_writer is not None:
        tb_writer.add_scalars(model_id, {'L-Train': train_stat['loss'],
                                         'L-Val': val_stat['loss']}, epoch)


def save_best_model(proj: Project, best_metric, logger, epoch, dev_stat, score_val, test_stat):
    if proj.save_every_epoch:  # Save every epoch
        best_metric = 0
        best_epoch = epoch
        logger.write_log(append=False, logfile=proj.logfile_best)
        torch.save(proj.net.state_dict(), proj.save_file)  # Save best PyTorch Model
        np.save(proj.save_file.replace('pt', 'npy'), test_stat['CONFUSION_MATRIX'])
        print('>>> saving best model from epoch %d to %s' % (epoch, proj.save_file))
    else:
        if score_val:
            if epoch == 0 or dev_stat['F1_SCORE_MICRO'] > best_metric:
                best_metric = dev_stat['F1_SCORE_MICRO']
                np.save(proj.save_file.replace('pt', 'npy'), dev_stat['CONFUSION_MATRIX'])
                logger.write_log(append=False, logfile=proj.logfile_best)
                torch.save(proj.net.state_dict(), proj.save_file)
                best_epoch = epoch
                print('>>> saving best model from epoch %d to %s' % (epoch, proj.save_file))
        else:
            if epoch == 0 or dev_stat['LOSS'] < best_metric:
                best_metric = dev_stat['LOSS']
                torch.save(proj.net.state_dict(), proj.save_file)
                logger.write_log(append=False, logfile=proj.logfile_best)
                best_epoch = epoch
                print('>>> saving best model from epoch %d to %s' % (epoch, proj.save_file))
    print("Best Metric: ", best_metric)
    return best_metric


def print_log(proj: Project, log_stat, train_stat, val_stat, test_stat, **kwargs):
    str_print = f"Epoch: {log_stat['EPOCH'] + 1:3d} of {log_stat['N_EPOCH']:3d} " \
                f"| Time: {log_stat['TIME_CURR']:s} " \
                f"| LR: {log_stat['LR_CURR']:1.5f} "
    for k, v in log_stat.items():
        if 'SP_' in k:
            str_print += "| " + k + f" {v * 100:3.2f}% "
    str_print += '\n'

    if train_stat is not None:
        str_print += f"    | Train-Loss: {log_stat['TRAIN_LOSS']:4.3f} " \
                     f"| Train-Reg: {log_stat['TRAIN_REG']:4.2f} |\n"
    # Validation
    if 'VAL_LOSS' in log_stat.keys():
        str_print += f"    | Val-Loss: {log_stat['VAL_LOSS']:4.3f}"
    if 'VAL_F1_SCORE_MICRO' in log_stat.keys():
        str_print += f" | Val-ACC: {log_stat['VAL_F1_SCORE_MICRO'] * 100:3.3f}% "
    if 'VAL_SP_T' in log_stat.keys():
        str_print += f"| Val-SP_T_DX: {log_stat['VAL_SP_T_DX'] * 100:3.2f}% " \
                     f"| Val-SP_T_DH: {log_stat['VAL_SP_T_DH'] * 100:3.2f}% " \
                     f"| Val-SP_T_DV: {log_stat['VAL_SP_T_DV'] * 100:3.2f}% |"
    str_print += '\n'
    # Test
    if 'TEST_LOSS' in log_stat.keys():
        str_print += f"    | Test-Loss: {log_stat['TEST_LOSS']:4.3f}"
    if 'TEST_F1_SCORE_MICRO' in log_stat.keys():
        str_print += f" | Test-ACC: {log_stat['TEST_F1_SCORE_MICRO'] * 100:3.3f}% "
    if 'TEST_SP_T' in log_stat.keys():
        str_print += f"| Test-SP_T_DX: {log_stat['TEST_SP_T_DX'] * 100:3.2f}% " \
                     f"| Test-SP_T_DH: {log_stat['TEST_SP_T_DH'] * 100:3.2f}% " \
                     f"| Test-SP_T_DV: {log_stat['TEST_SP_T_DV'] * 100:3.2f}% |"
        str_print += '\n'
    print(str_print)
    # Other Info
    for k, v in kwargs.items():
        print(k + "={x:.2f}".format(x=v), end=' | ')
    print("\n")


def gen_log_stat(proj: Project, epoch, start_time, train_stat=None, val_stat=None, test_stat=None):
    # End Timer
    time_curr = util.timeSince(start_time)

    # Get current learning rate
    lr_curr = 0
    if proj.optimizer is not None:
        for param_group in proj.optimizer.param_groups:
            lr_curr = param_group['lr']

    # Create log dictionary
    log_stat = {'SEED': proj.seed,
                'EPOCH': epoch,
                'N_EPOCH': proj.n_epochs,
                'TIME_CURR': time_curr,
                'BATCH_SIZE': proj.batch_size,
                'NUM_PARAMS': proj.num_params,
                'LOSS_FUNC': proj.loss,
                'OPT': proj.opt,
                'LR_CURR': lr_curr,
                'HID_DROPOUT': proj.rnn_dropout
                }

    # Merge stat dicts into the log dict
    if train_stat is not None:
        train_stat_log = {f'TRAIN_{k.upper()}': v for k, v in train_stat.items()}
        log_stat = {**log_stat, **train_stat_log}
    if val_stat is not None:
        val_stat_log = {f'VAL_{k.upper()}': v for k, v in val_stat.items()}
        del val_stat_log['VAL_LR_CRITERION']
        log_stat = {**log_stat, **val_stat_log}
    if test_stat is not None:
        test_stat_log = {f'TEST_{k.upper()}': v for k, v in test_stat.items()}
        del test_stat_log['TEST_LR_CRITERION']
        log_stat = {**log_stat, **test_stat_log}

    # Evaluate Classifier Weight Sparsity
    log_stat.update(proj.net.get_sparsity())
    # if proj.model_name != 'FC':
    #     n_nonzero_weight_elem = 0
    #     n_weight_elem = 0
    #     for name, param in proj.net.named_parameters():
    #         if 'rnn' in name:
    #             if 'weight' in name:
    #                 n_nonzero_weight_elem += len(torch.nonzero(param.data))
    #                 n_weight_elem += param.data.nelement()
    #     log_stat['SP_W_CLA'] = 1 - (n_nonzero_weight_elem / n_weight_elem)

    # Evaluate FC Extra Layer Weight Sparsity
    # log_stat['SP_W_FC'] = 0
    # if proj.fc_extra_size:
    #     n_nonzero_weight_elem = 0
    #     n_weight_elem = 0
    #     for name, param in proj.net.named_parameters():
    #         if 'fc_extra' in name:
    #             if 'weight' in name:
    #                 n_nonzero_weight_elem += len(torch.nonzero(param.data))
    #                 n_weight_elem += param.data.nelement()
    #     log_stat['SP_W_FC'] = 1 - (n_nonzero_weight_elem / n_weight_elem)
    return log_stat
