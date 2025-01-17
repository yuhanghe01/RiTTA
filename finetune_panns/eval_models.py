"""
Evaluate Audio Event Detection Model
"""
import torch
from torch.utils.data import DataLoader
import yaml
import os
from scipy.io import wavfile
import models
import DataProvider
from sklearn import metrics
import numpy as np

class Evaluator(object):
    def __init__(self):
        pass

    def append_to_dict(self, input_dict, key, value):
        if key in input_dict.keys():
            input_dict[key].append(value)
        else:
            input_dict[key] = [value]


    def eval(self, pretrained_model_path: str=None, 
             args: dict=None, 
             device: str=None):
        # dataset dataloader
        dataset = DataProvider.PANNSTaggingData(config=args)
        data_loader = DataLoader(dataset=dataset,
                                batch_size=args['TRAIN_CONFIG']['batch_size'],
                                shuffle=True,
                                drop_last=True,
                                num_workers=8)
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
        output_dict = {'clipwise_output': [], 'target': []}
        for idx, data in enumerate(data_loader):
            input_waveform, input_label = data[0], data[1]
            input_waveform = input_waveform.to(device)
            input_label = input_label.to(device)

            logits = model(input_waveform)
            score = torch.nn.functional.sigmoid(logits)
            score = score.cpu().detach().numpy()
            input_label = input_label.cpu().detach().numpy()

            output_dict['clipwise_output'].append(score)
            output_dict['target'].append(input_label)

        clipwise_output = output_dict['clipwise_output']    # (audios_num, classes_num)
        target = output_dict['target']    # (audios_num, classes_num)
        clipwise_output = np.concatenate(clipwise_output, axis=0)
        target = np.concatenate(target, axis=0)

        clipwise_output = clipwise_output.reshape(-1, args['DATA_CREATION_CONFIG']['class_num'])
        target = target.reshape(-1, args['DATA_CREATION_CONFIG']['class_num'])

        average_precision = metrics.average_precision_score(
            target, clipwise_output, average=None)

        auc = metrics.roc_auc_score(target, clipwise_output, average=None)
        
        statistics = {'average_precision': average_precision, 'auc': auc}

        mAP = np.mean(average_precision)
        mAUC = np.mean(auc)

        return mAP, mAUC
    
def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    device = torch.device(dev)
    print("using device: ", dev)
    model_id_list = np.arange(100,550,50).tolist()
    mAP_list = []
    mAUC_list = []
    for model_id in model_id_list:    
        evaluator = Evaluator()
        pretrained_model = 'model_epoch_{}.pth'.format(model_id)
        assert os.path.exists(pretrained_model)
        mAP, mAUC = evaluator.eval(pretrained_model_path=pretrained_model, 
                    args=args,
                    device=device)
        mAP_list.append(mAP)
        mAUC_list.append(mAUC)

    print("mAP: ", mAP_list)
    print("mAUC: ", mAUC_list)

    #get the best-performing model
    best_model_id = model_id_list[np.argmax(mAP_list)]
    print("Best model id: ", best_model_id)

if __name__ == "__main__":
    yaml_config_filename = 'config.yaml'
    assert os.path.exists(yaml_config_filename)
    with open(yaml_config_filename) as f:
        args = yaml.safe_load(f)

    main(args)
