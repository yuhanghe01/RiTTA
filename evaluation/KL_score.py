import numpy as np
import os
from scipy import linalg
import glob
import pickle
from numpy.linalg import inv, det

class KLDistance:
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

    def kl_divergence(self, mu1, sigma1, mu2, sigma2):
        """Compute KL divergence between two multivariate Gaussians."""
        sigma2_inv = inv(sigma2)
        mu_diff = mu2 - mu1

        # Compute the terms of the KL divergence
        trace_term = np.trace(sigma2_inv @ sigma1)
        mean_term = mu_diff.T @ sigma2_inv @ mu_diff
        det_term = np.log(det(sigma2) / det(sigma1))
        d = mu1.shape[0]

        # Final KL divergence value
        return 0.5 * (trace_term + mean_term - d + det_term)

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

    def get_KL_score(self):
        # background_dir: generated samples
        # eval_dir: groundtruth samples
        # embed_ref = self.get_embeddings(self.reference_dir)
        # embed_pred = self.get_embeddings(self.pred_dir)
        embed_ref, embed_pred = self.get_ref_pred_embeddings()
        mu_ref, sigma_ref = self.calculate_embd_statistics(embed_ref)
        mu_pred, sigma_pred = self.calculate_embd_statistics(embed_pred)
        fad_score = self.kl_divergence(mu_ref, sigma_ref, mu_pred, sigma_pred)

        return fad_score

if __name__ == "__main__":
    pred_dir = '/mnt/nas/yuhang/audioldm/audioldm_data' # 38.94636692654581
    pred_dir = '/mnt/nas/yuhang/audioldm/tango2_data' # 89.66388428005241
    pred_dir = '/mnt/nas/yuhang/audioldm/tango_data' # 90.2618047456912
    # pred_dir = '/mnt/nas/yuhang/audioldm/makeanaudio_data' # 82.7153162546361
    # pred_dir = '/mnt/nas/yuhang/audioldm/audiogen_data' # vggish: 28.018734886740397
    pred_dir = '/mnt/nas/yuhang/audioldm/audioldm_LFull_data' # 38.42170356697003
    pred_dir = '/mnt/nas/yuhang/audioldm/audioldm2_LFull_data' # 29.074793744929032
    pred_dir = '/mnt/nas/yuhang/audioldm/tango2_finetune' # 38.94636692654581
    pred_dir = '/mnt/nas/yuhang/audioldm/tango-finetuned' # 38.94636692654581
    ref_dir = '/mnt/nas/yuhang/audioldm/gen_data_bak'
    data_dict_filename = os.path.join(ref_dir, 'data_dict.pkl')
    kl_scoring = KLDistance(ref_dir, pred_dir, data_dict_filename=data_dict_filename,
                               use_panns_embed=False, TTA_method_name='tango')
    kl_score = kl_scoring.get_KL_score()
    print(kl_score)