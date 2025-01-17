"""
Evaluate Audio Event Detection Model
"""
import torch
import yaml
import os
import models
import numpy as np
import glob
import librosa

class Evaluator(object):
    def __init__(self):
        pass

    def append_to_dict(self, input_dict, key, value):
        if key in input_dict.keys():
            input_dict[key].append(value)
        else:
            input_dict[key] = [value]


    def eval(self, pretrained_model_path: str=None,
             audio_dir: str=None,
             args: dict=None, 
             device: str=None):
        # dataset dataloader
        # dataset = DataProvider.PANNSTaggingData(config=args)
        # data_loader = DataLoader(dataset=dataset,
        #                         batch_size=args['TRAIN_CONFIG']['batch_size'],
        #                         shuffle=True,
        #                         drop_last=True,
        #                         num_workers=8)
        model = models.Cnn14_DecisionLevelMax(sample_rate=args['DATA_CREATION_CONFIG']['sample_rate'],
                                window_size=args['DATA_CREATION_CONFIG']['window_size'], 
                                hop_size=args['DATA_CREATION_CONFIG']['hop_size'], 
                                mel_bins=args['DATA_CREATION_CONFIG']['mel_bins'], 
                                fmin=args['DATA_CREATION_CONFIG']['fmin'],
                                fmax=args['DATA_CREATION_CONFIG']['fmax'],
                                classes_num=args['DATA_CREATION_CONFIG']['class_num'])
        # load the pretrained model
        pretrained_model = torch.load(pretrained_model_path)

        model_state = model.state_dict()
        model_state.update(pretrained_model['model'])
        model.load_state_dict(model_state)

        model = model.to(device=device)
        model = model.eval()
        audio_files = glob.glob(os.path.join(audio_dir, '*.wav'))
        for audio_id, audio_file in enumerate(audio_files):
            if audio_id % 100 == 0:
                print('Processing audio file: {}/{}'.format(audio_id, len(audio_files)))
            audio_wave = librosa.load(audio_file, sr=args['DATA_CREATION_CONFIG']['sample_rate'])[0]
            audio_wave = torch.tensor(audio_wave).unsqueeze(0)
            input_waveform = audio_wave.to(device)

            logits = model(input_waveform)
            score = torch.sigmoid(logits)
            score = score.cpu().detach().numpy().squeeze()

            save_filename = os.path.basename(audio_file).replace('.wav', '_panns_det.npy')
            np.save(os.path.join(audio_dir, save_filename), score)
    
def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    device = torch.device(dev)
    print("using device: ", dev)  
    evaluator = Evaluator()
    pretrained_model = 'model.pth'
    audio_dir = 'audioldm_data'
    audio_dir = 'audiogen_data'
    audio_dir = 'makeanaudio_data'
    audio_dir = 'tango_data'
    audio_dir = 'tango2_data'
    audio_dir = 'audioldm_LFull_data'
    audio_dir = 'audioldm2_LFull_data'
    audio_dir = 'tango2_finetune'
    audio_dir = 'tango-finetuned'
    assert os.path.exists(pretrained_model)
    evaluator.eval(pretrained_model_path=pretrained_model, 
                   audio_dir=audio_dir,
                   args=args,
                   device=device)

if __name__ == "__main__":
    yaml_config_filename = 'config.yaml'
    assert os.path.exists(yaml_config_filename)
    with open(yaml_config_filename) as f:
        args = yaml.safe_load(f)

    main(args)
