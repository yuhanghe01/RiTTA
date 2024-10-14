import numpy as np
import os
import torch
import torch.nn as nn
import pann_models
import librosa
import glob

class EmbedExtractor:
    def __init__(self, config = None) -> None:
        self.config = config
        assert len(self.audio_filename_list) > 0
        self.get_vggish_model()

    def get_vggish_model(self):
        use_pca = self.config['vggish_config']['use_pca']
        use_activation = self.config['vggish_config']['use_activation']
        model = torch.hub.load("harritaylor/torchvggish", "vggish")
        if use_pca:
            model.postprocess = False
        if not use_activation:
            model.embeddings = nn.Sequential(*list(model.embeddings.children())[:-1])
        model.postprocess = False
        model.embeddings = nn.Sequential(
            *list(model.embeddings.children())[:-1])
        model.eval()

        self.vggish_model = model

    def get_panns_model(self):
        '''use the panns pretrained audiotagging model'''
        sample_rate = self.config['panns_config']['sample_rate']
        window_size = self.config['panns_config']['window_size']
        hop_size = self.config['panns_config']['hop_size']
        mel_bins = self.config['panns_config']['mel_bins']
        fmin = self.config['panns_config']['fmin']
        fmax = self.config['panns_config']['fmax']
        model = pann_models.Cnn14(sample_rate=sample_rate, 
                                window_size=window_size, 
                                hop_size=hop_size, 
                                mel_bins=mel_bins, 
                                fmin=fmin, 
                                fmax=fmax, 
                                classes_num=527)
        checkpoint_path = self.config['PANNS_cptpath']
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model.eval()
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)

        self.panns_model = model

    def get_vggish_embed(self, audio_filename_list):
        for audio_id, audio_filename in enumerate(audio_filename_list):
            if audio_id % 100 == 0:
                print(f'Processing {audio_id}/{len(audio_filename_list)}')
            (waveform, _) = librosa.core.load(audio_filename, sr=16000, mono=True)

            embed = self.vggish_model.forward(waveform, 16000)
            embed = embed.cpu().detach().numpy().squeeze()
            embed_filename = audio_filename.replace('.wav', '_vggish_embed.npy')
            np.save(embed_filename, embed)

    def get_panns_embed(self, audio_filename_list):
        for audio_id, audio_filename in enumerate(audio_filename_list):
            if audio_id % 100 == 0:
                print(f'Processing {audio_id}/{len(audio_filename_list)}')
            (waveform, _) = librosa.core.load(audio_filename, sr=16000, mono=True)
            waveform = torch.from_numpy(waveform).to(torch.float32).to(device)
            waveform = waveform.unsqueeze(0)

            output_dict = self.panns_model(waveform, None)
            embed = output_dict['embedding'].detach().cpu().numpy().squeeze()

            embed_filename = audio_filename.replace('.wav', '_panns_embed.npy')
            np.save(embed_filename, embed)

    def get_embedding(self, audio_dir, embed_type = ['vggish', 'panns']):
        audio_filename_list = glob.glob(os.path.join(audio_dir, '*.wav'))
        if 'vggish' in embed_type:
            self.get_vggish_embed(audio_filename_list)
        if 'panns' in embed_type:
            self.get_panns_embed(audio_filename_list)
            
        print('Done')

