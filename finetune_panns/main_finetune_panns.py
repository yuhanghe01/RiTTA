import models
import torch
import os
import yaml
from torch.utils.data import DataLoader
import models
import DataProvider
from torch.utils.tensorboard import SummaryWriter

def load_from_pretrain(pretrained_checkpoint_path, model, freeze_base=True):
    checkpoint = torch.load(pretrained_checkpoint_path)
    model.load_state_dict(checkpoint['model'], strict=False)

    for name, param in model.named_parameters():
        print(name, param.requires_grad)
        if 'fc_audioset_audiollm' in name:
            param.requires_grad = True
        else:
            param.requires_grad = not freeze_base
            
    return model

def train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.cuda.manual_seed(42)
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    device = torch.device(dev)

    dataset = DataProvider.PANNSDetData(config=args)
    loss_func = torch.nn.BCELoss()

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


    if args['TRAIN_CONFIG']['finetune']:
        pretrained_model = args['TRAIN_CONFIG']['pretrained_model']
        model = load_from_pretrain(pretrained_model, 
                                   model, 
                                   freeze_base=args['TRAIN_CONFIG']['freeze_base'])
    model.train()
    model.to(device)

    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('total param = {}'.format(total_params))

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args['TRAIN_CONFIG']['init_lr'],
                                 betas=(0.9, 0.999),
                                 eps=1e-08,
                                 weight_decay=args['TRAIN_CONFIG']['weight_decay'],
                                 amsgrad=True)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=args['TRAIN_CONFIG']['lr_decay_epochs'],
                                                   gamma=args['TRAIN_CONFIG']['lr_decay_gamma'])

    writer = SummaryWriter(log_dir=args['TRAIN_CONFIG']['ckpt_save_dir'])

    print("\n=======  Launched Training  ======= \n")

    #step5: start to train
    for ep in range(1, args['TRAIN_CONFIG']['train_epochs'] + 1):
        for idx, data in enumerate(data_loader):
            input_waveform, input_label = data[0], data[1]
            input_waveform = input_waveform.to(device)
            input_label = input_label.to(device)

            logits = model(input_waveform)

            loss = loss_func(torch.sigmoid(logits), 
                             input_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_lr = optimizer.param_groups[0]['lr']
            if idx % 10 == 0:
                print("== step: [{}/{}] | epoch: [{}/{}] | loss: {:.3f} | lr = {}".format(idx + 1,
                                                                                 len(data_loader),
                                                                                 ep,
                                                                                 args['TRAIN_CONFIG']['train_epochs'],
                                                                                 loss.detach().cpu().numpy(),
                                                                                 current_lr))
                
                writer.add_scalar('Loss/train', loss.detach().cpu().numpy(),  ep*len(data_loader) + idx)
                writer.add_scalar('LearnRate/', current_lr)

        if ep % args['TRAIN_CONFIG']['save_every_n_epochs'] == 0:
            model_save_basename = 'TRC_model_epoch_{}.pth'.format(ep)
            outdict = {
                'epoch': ep,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),}

            torch.save(outdict, os.path.join(args['TRAIN_CONFIG']['ckpt_save_dir'], model_save_basename))

        lr_scheduler.step()

    writer.close()
    print("\n=======  Training Finished  ======= \n")

if __name__ == "__main__":
    yaml_config_filename = 'finetune_config.yaml'
    assert os.path.exists(yaml_config_filename)
    with open(yaml_config_filename) as f:
        args = yaml.safe_load(f)

    train(args)