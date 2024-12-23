import numpy as np
import os
import torch
import torch.nn as nn
from scipy import linalg
from tqdm import tqdm
import soundfile as sf
import glob
import pickle

class FrechetAudioDistance:
    def __init__(self, reference_dir, pred_dir, use_panns_embed=True, data_dict_filename=None,
                 TTA_method_name = 'audioldm' ):
        self.reference_dir = reference_dir
        self.pred_dir = pred_dir
        self.use_panns_embed = use_panns_embed
        self.TTA_method_name = TTA_method_name
        self.feat_embed_name = '_panns_embed.npy' if use_panns_embed else '_vggish_embed.npy'

        with open(data_dict_filename, 'rb') as f:
            self.data_dict = pickle.load(f)
    def get_ref_pred_embeddings(self, main_cate_name = None):
        ref_embed_list = list()
        pred_embed_list = list()

        for main_cate in self.data_dict.keys():
            if main_cate in ['time', 'author']:
                continue
            if main_cate_name is not None and main_cate != main_cate_name:
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
                    # pred_embed_list.append(ref_embed)

        """return np.concatenate(ref_embed_list, axis=0), np.concatenate(pred_embed_list, axis=0)"""
        return np.concatenate(ref_embed_list, axis=0), np.concatenate(pred_embed_list, axis=0)

    def get_embeddings(self, input_dir):
        """
        Get embeddings using VGGish model.
        Params:
        -- x    : Either
            (i) a string which is the directory of a set of audio files, or
            (ii) a list of np.ndarray audio samples
        -- sr   : Sampling rate, if x is a list of audio samples. Default value is 16000.
        """
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

        return np.concatenate(embed_list, axis=0)

    def calculate_embd_statistics(self, embd_lst):
        if isinstance(embd_lst, list):
            embd_lst = np.array(embd_lst)
        mu = np.mean(embd_lst, axis=0)
        sigma = np.cov(embd_lst, rowvar=False)

        return mu, sigma

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """
        Adapted from: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
        Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                inception net (like returned by the function 'get_predictions')
                for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                representative data set.
        Returns:
        --   : The Frechet Distance.
        """
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert (
            mu1.shape == mu2.shape
        ), "Training and test mean vectors have different lengths"
        assert (
            sigma1.shape == sigma2.shape
        ), "Training and test covariances have different dimensions"

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = (
                "fid calculation produces singular product; "
                "adding %s to diagonal of cov estimates"
            ) % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    def get_fad_score(self):
        # background_dir: generated samples
        # eval_dir: groundtruth samples
        # embed_ref = self.get_embeddings(self.reference_dir)
        # embed_pred = self.get_embeddings(self.pred_dir)
        embed_ref, embed_pred = self.get_ref_pred_embeddings()
        mu_ref, sigma_ref = self.calculate_embd_statistics(embed_ref)
        mu_pred, sigma_pred = self.calculate_embd_statistics(embed_pred)
        fad_score = self.calculate_frechet_distance(mu_ref, sigma_ref, mu_pred, sigma_pred)

        return fad_score

    def get_fad_score_wrt_maincate(self):
        maincate_list = ['Count', 'Temporal_Order', 'Spatial_Distance', 'Compositionality']
        for main_cate in maincate_list:
            embed_ref, embed_pred = self.get_ref_pred_embeddings(main_cate)
            mu_ref, sigma_ref = self.calculate_embd_statistics(embed_ref)
            mu_pred, sigma_pred = self.calculate_embd_statistics(embed_pred)
            fad_score = self.calculate_frechet_distance(mu_ref, sigma_ref, mu_pred, sigma_pred)
            # print('Main cate: ', main_cate)
            print('main_cate: {}, FAD score: {}'.format(main_cate,fad_score))

if __name__ == "__main__":
    pred_dir = '/mnt/nas/yuhang/audioldm/audioldm_data' #5.65298179707108
    # pred_dir = '/mnt/nas/yuhang/audioldm/tango2_data' # 13.83934504194481
    # pred_dir = '/mnt/nas/yuhang/audioldm/tango_data' # 10.792733585616613
    # pred_dir = '/mnt/nas/yuhang/audioldm/makeanaudio_data' # 9.461433261937735
    # pred_dir = '/mnt/nas/yuhang/audioldm/audiogen_data' # 6.4340542277877475
    # pred_dir = '/mnt/nas/yuhang/audioldm/audioldm_LFull_data' # 5.468262784257881
    pred_dir = '/mnt/nas/yuhang/audioldm/audioldm2_LFull_data' # 6.687262388865676
    pred_dir = '/mnt/nas/yuhang/audioldm/tango2_finetune' # 10.668489915436098
    pred_dir = '/mnt/nas/yuhang/audioldm/tango-finetuned' # 10.668489915436098
    ref_dir = '/mnt/nas/yuhang/audioldm/gen_data_bak'
    data_dict_filename = os.path.join(ref_dir, 'data_dict.pkl')
    fad = FrechetAudioDistance(ref_dir, pred_dir, data_dict_filename=data_dict_filename,
                               use_panns_embed=False, TTA_method_name='tango')
    # fad_score = fad.get_fad_score_wrt_maincate()
    fad_score = fad.get_fad_score()
    print(fad_score)