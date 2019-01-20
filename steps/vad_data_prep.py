import os
import sys
import numpy as np
from glob import glob
import pandas as pd
from tqdm import tqdm


class QUTTIMITDataPrepare:
    def __init__(self, path_dataset, path_output):
        # Data Path
        self.path_dataset = path_dataset
        self.path_output = path_output

        # Create a dict for label to index conversion
        self.labels = pd.read_csv(os.path.join(self.path_output, 'labels.csv'))
        self.dict_label2int61 = {}
        self.dict_label2int48 = {}
        self.dict_label2int39 = {}
        for idx, x in enumerate(self.labels['label61']):
            self.dict_label2int61[x] = idx
        for idx, x in enumerate(self.labels['label48']):
            self.dict_label2int48[x] = idx
        for idx, x in enumerate(self.labels['label39']):
            self.dict_label2int39[x] = idx

    def phn_text_to_int_str(self, str_phn_text, dict_label2int):
        list_phn_text = str_phn_text.split()
        list_int_str = [int(dict_label2int[x]) for x in list_phn_text]
        # int_str = ' '.join(list_int_str)
        return list_int_str

    def create_qut_label(self, filename):
        with open(filename + '.eventlab') as f:
            row = f.readlines()
            timestep = []
            label = []
            for x in row:
                timestep.append(float(x.strip().split(' ')[1]))
                label_curr = x.strip().split(' ')[2]
                if label_curr == 'speech':
                    label.append(1)
                else:
                    label.append(0)
            return timestep, label

    def collect_dataset(self):
        # Standard parameters
        columns = ['group', 'category', 'modality', 'snr', 'key', 'path', 'timestep', 'label']

        # Get data folder
        all_data = os.path.join(self.path_dataset, 'QUT-NOISE-TIMIT')

        # Show Dataset Path
        # print("Train Data Path: ", train)
        # print("Test Data Path: ", test)

        # Get group A lists
        with open(os.path.join(self.path_output, 'qut_group_a.txt')) as f:
            list_group_a = f.readlines()
            list_group_a = [x.strip() for x in list_group_a]

        # Loop over folders
        description_list = []
        for group, folder in zip(['all_data'], [all_data]):
            all_file_paths = [y for x in os.walk(folder) for y in glob(os.path.join(x[0], '*.wav'))]
            all_file_paths = sorted(all_file_paths)
            key_idx = 0
            for idx, file_path in tqdm(enumerate(all_file_paths)):
                file_name = file_path.split('/')[-1].split('.')[0]
                category = file_name.split('_')[0]
                snr = file_name.split('_')[3]

                key_idx += 1

                label_file_path = file_path[0:-4]
                timestep, label = self.create_qut_label(label_file_path)

                row = {}
                if category in list_group_a:
                    row['group'] = 'a'
                else:
                    row['group'] = 'b'
                row['category'] = category
                row['modality'] = 'audio'
                row['snr'] = snr
                row['key'] = '{}_{}'.format(row['group'], key_idx)
                row['path'] = file_path
                row['timestep'] = timestep
                row['label'] = label

                description_list.append(row)
        df = pd.DataFrame(description_list, columns=columns)
        description_file = os.path.join(self.path_output, 'description_qut.csv')
        df.to_csv(description_file, index=False)


def main(path_dataset, proj_root):
    print("####################################################################")
    print("# VAD Step 0: Data Preparation                                     #")
    print("####################################################################")
    np.random.seed(0)
    path_output = os.path.join(proj_root, 'data')
    print("QUT-NOISE-TIMIT Dataset Root Path: ", path_dataset)
    print("Output Path:             ", path_output)

    data_prepare = QUTTIMITDataPrepare(path_dataset, path_output)
    data_prepare.collect_dataset()

    print("Data Preparation Completed...                                       ")
    print(" ")
