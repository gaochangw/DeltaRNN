import os
import sys
import numpy as np
from glob import glob
import pandas as pd
from tqdm import tqdm


class TIMITDataPrepare:
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

    def map_phn_text(self, str_phn_text, type, path_target):
        phn_map_61_48_39 = pd.read_csv(os.path.join(path_target, 'phn_map_61_48_39.csv'))
        dict_61_48 = {}
        dict_61_39 = {}
        for idx, x in enumerate(phn_map_61_48_39['phn-61']):
            dict_61_48[x] = phn_map_61_48_39['phn-48'][idx]  # Create PHN-61 to PHN-48 conversion dict
            dict_61_39[x] = phn_map_61_48_39['phn-39'][idx]  # Create PHN-61 to PHN-39 conversion dict

        phn_map_48_39 = pd.read_csv(os.path.join(path_target, 'phn_map_48_39.csv'))
        dict_48_39 = {}
        for idx, x in enumerate(phn_map_48_39['phn-48']):
            dict_48_39[x] = phn_map_48_39['phn-39'][idx]  # Create PHN-48 to PHN-39 conversion dict

        list_phn_text = str_phn_text.split()
        if type == 1:  # Type 1 - PHN--61 to PHN-48 conversion
            list_mapped_phn_text = [dict_61_48[x] for x in list_phn_text]
        elif type == 2:  # Type 2 - PHN--61 to PHN-39 conversion
            list_mapped_phn_text = [dict_61_39[x] for x in list_phn_text]
        elif type == 3:  # Type 3 - PHN--48 to PHN-39 conversion
            list_mapped_phn_text = [dict_48_39[x] for x in list_phn_text]
        else:
            print("ERROR: Only type 1~3 are supportted...")
            sys.exit(1)

        # Remove space characters
        while ' ' in list_mapped_phn_text:
            list_mapped_phn_text.remove(' ')

        # Convert list of phone text into a single string seperated by space chars
        str_mapped_phn_text = ' '.join(list_mapped_phn_text)

        return str_mapped_phn_text


    def collect_dataset(self):
        # Standard parameters
        columns = ['group', 'uttid', 'modality', 'key', 'path', 'label61', 'label48', 'label39', 'phn61', 'phn48', 'phn39']

        # Get train and test folders
        train = os.path.join(self.path_dataset, 'TRAIN')
        test = os.path.join(self.path_dataset, 'TEST')

        # Show Dataset Path
        # print("Train Data Path: ", train)
        # print("Test Data Path: ", test)

        # Get core test set according to lists
        with open(os.path.join(self.path_output, 'test_list.txt')) as f:
            list_test = f.readlines()
            list_test = [x.strip() for x in list_test]

        # Loop over folders
        description_list = []
        for group, folder in zip(['train', 'test'], [train, test]):
            all_file_paths = [y for x in os.walk(folder) for y in glob(os.path.join(x[0], '*.WAV'))]
            all_file_paths = sorted(all_file_paths)
            key_idx = 0
            for idx, file_path in tqdm(enumerate(all_file_paths)):
                file_name = file_path.split('/')[-1].split('.')[0]
                speaker_id = file_path.split('/')[-2]

                if (file_name.find('SA') == -1):  # Remove SA utterance from dataset
                    # Get utterance phone label text
                    with open(file_path.split('.')[0] + '.PHN') as f:
                        transcript = f.readlines()
                        transcript = [x.strip() for x in transcript]
                        phn61 = [x.split(' ')[-1] for x in transcript]
                        phn61 = ' '.join(phn61)

                    # Map PHN-61 to PHN-48 & PHN-39
                    phn48 = self.map_phn_text(phn61, 1, self.path_output)
                    phn39 = self.map_phn_text(phn61, 2, self.path_output)

                    # Get label int strings
                    label61 = self.phn_text_to_int_str(phn61, self.dict_label2int61)
                    label48 = self.phn_text_to_int_str(phn48, self.dict_label2int48)
                    label39 = self.phn_text_to_int_str(phn39, self.dict_label2int39)

                    key_idx += 1
                    row = {}
                    if group == 'train':
                        row['group'] = 'train'
                    else:
                        if speaker_id in list_test:
                            row['group'] = 'test'
                        else:
                            row['group'] = 'dev'
                    row['uttid'] = speaker_id + '_' + file_name
                    row['modality'] = 'audio'
                    row['key'] = '{}_{}'.format(row['group'], key_idx)
                    row['path'] = file_path
                    row['label61'] = label61
                    row['label48'] = label48
                    row['label39'] = label39
                    row['phn61'] = phn61
                    row['phn48'] = phn48
                    row['phn39'] = phn39
                    description_list.append(row)
        df = pd.DataFrame(description_list, columns=columns)
        description_file = os.path.join(self.path_output, 'description.csv')
        df.to_csv(description_file, index=False)


def main(path_dataset, proj_root):
    print("####################################################################")
    print("# ASR Step 0: Data Preparation                                     #")
    print("####################################################################")
    np.random.seed(0)
    path_output = os.path.join(proj_root, 'data')
    print("TIMIT Dataset Root Path: ", path_dataset)
    print("Output Path:             ", path_output)
    
    data_prepare = TIMITDataPrepare(path_dataset, path_output)
    data_prepare.collect_dataset()

    print("Data Preparation Completed...                                       ")
    print(" ")
