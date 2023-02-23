import os
from project import Project


def main(proj: Project):
    # Dataset Paths
    dataset_name = 'speech_commands_v0.02'
    testset_name = 'speech_commands_test_set_v0.02'
    dataset_path = os.path.join(proj.data_dir, dataset_name)
    testset_path = os.path.join(proj.data_dir, testset_name)
    output_path = os.path.join('data', proj.dataset)

    # Data Augmentation
    all_speeds = [0.9, 1.1]  # speed variations
    list_snr = [10, 5]  # noise levels

    prepare = proj.data_prep.DataPrepare(proj=proj,
                                         dataset_path=dataset_path,
                                         testset_path=testset_path,
                                         output_path=output_path)

    # Augment
    try:
        if proj.augment_noise:
            prepare.augment_noise(list_snr)
    except AttributeError:
        pass

    # Creat Silence Samples
    try:
        prepare.create_silence()
    except AttributeError:
        pass

    # Collect Dataset
    try:
        if proj.augment_noise:
            for target_snr in list_snr:
                prepare.collect(augment_noise=1, target_snr=target_snr)
        else:
            prepare.collect(augment_noise=0, target_snr=0)
    except AttributeError:
        prepare.collect(augment_noise=0, target_snr=0)
        pass


def gen_meter_args(args, dataset_path, output_path):
    dict_meter_args = {'dataset_path': dataset_path,
                       'output_path': output_path,
                       'n_targets': args.n_targets}
    if args.dataset_name == 'timit':
        dict_meter_args['phn'] = args.phn
    return dict_meter_args
