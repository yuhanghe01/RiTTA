import os
import torch
from torch.utils.data import Dataset
import numpy as np
import random
import scipy.io.wavfile as wavfile

class PANNSDetData(Dataset):
    def __init__(self, config):
        super(PANNSDetData, self).__init__()
        self.config = config
        self.sample_rate = self.config['DATA_CREATION_CONFIG']['sample_rate']
        self.get_label2idx()
        self.get_all_audio_files()
        if self.config['TRAIN_CONFIG']['mode'] == 'train':
            random.seed(self.config['TRAIN_CONFIG']['random_seeds'][0])
            np.random.seed(self.config['TRAIN_CONFIG']['random_seeds'][0])
        else:
            random.seed(self.config['TRAIN_CONFIG']['random_seeds'][1])
            np.random.seed(self.config['TRAIN_CONFIG']['random_seeds'][1])
        self.construct_audio_filenames()

    def prepare_data(self, audio_filename_list):
        audio_data_combined = np.zeros((self.config['DATA_CREATION_CONFIG']['sample_rate']*self.config['DATA_CREATION_CONFIG']['audio_len_sec']),
                              dtype=np.float32)
        sample_rate = self.config['DATA_CREATION_CONFIG']['sample_rate']
        audio_len_sec = self.config['DATA_CREATION_CONFIG']['audio_len_sec']
        time_resolution = self.config['DATA_CREATION_CONFIG']['time_resolution']
        class_num = self.config['DATA_CREATION_CONFIG']['class_num']
        label_len = int(audio_len_sec/time_resolution)
        audio_label_combined = np.zeros((label_len, class_num), 
                                        dtype=np.float32)  # need to use float32 for torch.nn.BCELoss
        # print(audio_filename_list)
        for audio_filename in audio_filename_list:
            audio_data = wavfile.read(audio_filename)[1].astype(np.float32)/32768.0
            audio_data_len = audio_data.shape[0]//self.config['DATA_CREATION_CONFIG']['sample_rate']
            max_sample_sec = audio_len_sec-audio_data_len
            start_sec = random.choice(np.arange(0, max_sample_sec+time_resolution, time_resolution).tolist())
            start_idx = int(start_sec*sample_rate)
            audio_data_combined[start_idx:start_idx+audio_data.shape[0]] += audio_data #linearly add all the audio data
            audio_category = audio_filename.split('/')[-2]
            audio_label = self.label_to_idx[audio_category]
            label_start_id = int(start_sec//time_resolution)
            label_end_id = label_start_id + int(audio_data_len//time_resolution)
            audio_label_combined[label_start_id:label_end_id,audio_label] = 1
            
        return audio_data_combined, audio_label_combined
    
    def construct_audio_filenames(self):
        self.audio_filename_list = list()
        if self.config['TRAIN_CONFIG']['mode'] == 'train':
            num2gen = self.config['TRAIN_CONFIG']['train_samples']
        else:
            num2gen = self.config['TRAIN_CONFIG']['test_samples']
        for num_id in range(num2gen):
            audio_num = np.random.randint( self.config['TRAIN_CONFIG']['min_polyphony'], self.config['TRAIN_CONFIG']['max_polyphony']+1)
            audio_filenames = list()
            for _ in range(audio_num):
                main_cate = random.choice(list(self.audio_dict.keys()))
                sub_cate = random.choice(list(self.audio_dict[main_cate].keys()))
                audio_filename_tmp = random.choice(self.audio_dict[main_cate][sub_cate])
                audio_filename_tmp = os.path.join(self.config['DATA_CREATION_CONFIG']['seed_audio_dir'], 
                                                  main_cate, 
                                                  sub_cate, 
                                                  audio_filename_tmp)
                assert os.path.exists(audio_filename_tmp)
                audio_filenames.append(audio_filename_tmp)

            self.audio_filename_list.append(audio_filenames)

        assert len(self.audio_filename_list) > 0

    def __len__(self):
        return len(self.audio_filename_list)

    def get_all_audio_files(self):
        self.audio_dict = dict()
        maincate_dirs = os.listdir(self.config['DATA_CREATION_CONFIG']['seed_audio_dir'])

        for maincate_dir in maincate_dirs:
            self.audio_dict[maincate_dir] = dict()
            subcate_dirs = os.listdir(os.path.join(self.config['DATA_CREATION_CONFIG']['seed_audio_dir'], 
                                                   maincate_dir))
            for subcate_dir in subcate_dirs:
                self.audio_dict[maincate_dir][subcate_dir] = []
                for audio_filename in os.listdir(os.path.join(self.config['DATA_CREATION_CONFIG']['seed_audio_dir'], 
                                                              maincate_dir, subcate_dir)):
                    if audio_filename.endswith('.wav'):
                        self.audio_dict[maincate_dir][subcate_dir].append(audio_filename)

    def get_label2idx(self):
        self.label_to_idx = dict()
        self.idx_to_label = dict()
        filename = self.config['DATA_CREATION_CONFIG']['label_filename']
        assert os.path.exists(filename)
        with open(filename, 'r') as f:
            for line in f.readlines():
                label, idx = line.strip('\t').split(' ')
                self.label_to_idx[label] = int(idx) - 1
                self.idx_to_label[int(idx)-1] = label
        
        self.classes_num = len(self.label_to_idx)

    def to_one_hot(self, k, classes_num):
        target = np.zeros(classes_num)
        target[k] = 1
        return target

    def pad_truncate_sequence(self, x, max_len):
        if len(x) < max_len:
            return np.concatenate((x, np.zeros(max_len - len(x))))
        else:
            return x[0 : max_len]

    def float32_to_int16(self, x):
        # assert np.max(np.abs(x)) <= 1.
        if np.max(np.abs(x)) > 1.:
            x /= np.max(np.abs(x))

        return (x * 32767.).astype(np.int16)
    
    def __getitem__(self, idx):
        audio_filenames = self.audio_filename_list[idx]
        audio_data, audio_label = self.prepare_data(audio_filenames)

        return torch.tensor(audio_data).to(torch.float32), torch.tensor(audio_label).to(torch.float32)