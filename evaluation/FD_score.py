import torch
import numpy as np
import scipy.linalg
from scipy.linalg import sqrtm
import pickle
import os
import glob

class FrechetDistance:
    def __init__(self, reference_dir, pred_dir, use_panns_embed=True, data_dict_filename=None,
                 TTA_method_name = 'audioldm' ):
        self.reference_dir = reference_dir
        self.pred_dir = pred_dir
        self.use_panns_embed = use_panns_embed
        self.TTA_method_name = TTA_method_name
        self.feat_embed_name = '_panns_embed.npy' if use_panns_embed else '_vggish_embed.npy'

        with open(data_dict_filename, 'rb') as f:
            self.data_dict = pickle.load(f)
    def get_ref_pred_embeddings(self):
        ref_embed_list = list()
        pred_embed_list = list()

        for main_cate in self.data_dict.keys():
            if main_cate in ['time', 'author']:
                continue
            for sub_cate in self.data_dict[main_cate].keys():
                if sub_cate == 'not':
                    continue
                # if sub_cate == 'f_then_else':
                for data_tmp in self.data_dict[main_cate][sub_cate]:
                    ref_audio_filename = data_tmp['reference_audio']
                    pred_audio_filename = data_tmp['reference_audio'].replace('.wav', '_{}.wav'.format(self.TTA_method_name))
                    assert os.path.exists(os.path.join(self.pred_dir, pred_audio_filename))
                    if sub_cate in ['if_then_else','or']:
                        ref_audio_filename1 = ref_audio_filename.replace('.wav', '_0.wav')
                        ref_audio_filename2 = ref_audio_filename.replace('.wav', '_1.wav')
                        assert os.path.exists(os.path.join(self.reference_dir, ref_audio_filename1))
                        assert os.path.exists(os.path.join(self.reference_dir, ref_audio_filename2))
                    else:
                        assert os.path.exists(os.path.exists(os.path.join(self.reference_dir, ref_audio_filename)))
                    pred_embed = np.load(os.path.join(self.pred_dir, pred_audio_filename.replace('.wav', self.feat_embed_name)))
                    if sub_cate in ['if_then_else','or']:
                        embed_filebasename1 = ref_audio_filename1.replace('.wav', self.feat_embed_name)
                        embed_filebasename2 = ref_audio_filename2.replace('.wav', self.feat_embed_name)
                        ref_embed1 = np.load(os.path.join(self.reference_dir, embed_filebasename1))
                        ref_embed2 = np.load(os.path.join(self.reference_dir, embed_filebasename2))
                        if np.sqrt(np.sum(np.square(pred_embed - ref_embed1 ))) < np.sqrt(np.sum(np.square(pred_embed - ref_embed2))):
                            ref_embed = ref_embed1
                        else:
                            ref_embed = ref_embed2
                    else:
                        ref_embed = np.load(os.path.join(self.reference_dir, ref_audio_filename.replace('.wav', self.feat_embed_name)))
                    ref_embed_list.append(ref_embed)
                    pred_embed_list.append(pred_embed)

        if self.use_panns_embed:
            return np.stack(ref_embed_list, axis=0), np.stack(pred_embed_list, axis=0)
        else:
            return np.concatenate(ref_embed_list, axis=0), np.concatenate(pred_embed_list, axis=0)

    def get_embeddings(self, input_dir):
        embed_list = list()
        if self.use_panns_embed:
            pattern_name = '*_panns_embed.npy'
        else:
            pattern_name = '*_vggish_embed.npy'
        embed_filename_list = glob.glob(os.path.join(input_dir, pattern_name))
        assert len(embed_filename_list) > 0, 'No embeddings found in {}'.format(input_dir)
        embed_filename_list = sorted(embed_filename_list)
        for embed_filename in embed_filename_list:
            embed = np.load(embed_filename).squeeze()
            embed_list.append(embed)

        if self.use_panns_embed:
            return np.stack(embed_list, axis=0)
        else:
            return np.concatenate(embed_list, axis=0)

    def calculate_embd_statistics(self, embd_lst):
        if isinstance(embd_lst, list):
            embd_lst = np.array(embd_lst)
        mu = np.mean(embd_lst, axis=0)
        sigma = np.cov(embd_lst, rowvar=False)

        return mu, sigma

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Calculate the Frechet Distance between two multivariate Gaussians."""
        mu_diff = mu1 - mu2
        covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
        
        # Numerical stability: Remove imaginary components if they exist
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        # Frechet Distance formula
        return np.sum(mu_diff**2) + np.trace(sigma1 + sigma2 - 2 * covmean)

    def get_fd_score(self):
        # background_dir: generated samples
        # eval_dir: groundtruth samples
        # embed_ref = self.get_embeddings(self.reference_dir)
        # embed_pred = self.get_embeddings(self.pred_dir)
        embed_ref, embed_pred = self.get_ref_pred_embeddings()
        mu_ref, sigma_ref = self.calculate_embd_statistics(embed_ref)
        mu_pred, sigma_pred = self.calculate_embd_statistics(embed_pred)
        fd_score = self.calculate_frechet_distance(mu_ref, sigma_ref, mu_pred, sigma_pred)

        return fd_score

if __name__ == "__main__":
    pred_dir = 'audioldm_data'
    data_dict_filename = os.path.join(ref_dir, 'data_dict.pkl')
    fd = FrechetDistance(ref_dir, pred_dir, data_dict_filename=data_dict_filename,
                               use_panns_embed=True, TTA_method_name='tango')
    fd_score = fd.get_fd_score()
    print(fd_score)
