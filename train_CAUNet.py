import torch
import torch.nn as nn
import numpy as np
import random
import os
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import json
import geomloss
from fastprogress import progress_bar
from argparse import ArgumentParser
from model.resnet_PADAA import wide_resnet50_2
from model.de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
from utils.utils_test import evaluation_multi_proj
from utils.SAKT import SAKTLayer, SCLoss, loss_fucntion
from dataset.FPAG_dataset import MVTecDataset_test, FPAG, get_data_transforms


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--save_folder', default='./result', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--detail_training', default='note', type=str)
    parser.add_argument('--proj_lr', default=0.001, type=float)
    parser.add_argument('--distill_lr', default=0.005, type=float)
    parser.add_argument('--weight_proj', default=0.05, type=float)
    parser.add_argument('--classes', nargs="+",
                        default=['carpet'])
    pars = parser.parse_args()
    return pars


def train(_class_, pars):
    print(_class_)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_transform, gt_transform = get_data_transforms(pars.image_size, pars.image_size)

    train_path = './data/mvtec/' + _class_ + '/train'
    test_path = './data/mvtec/' + _class_

    if not os.path.exists(pars.save_folder + '/' + _class_):
        os.makedirs(pars.save_folder + '/' + _class_)
    save_model_path = pars.save_folder + '/' + _class_ + '/' + 'wres50_' + _class_ + '.pth'

    train_data = FPAG(root=train_path, transform=data_transform)
    test_data = MVTecDataset_test(root=test_path, transform=data_transform, gt_transform=gt_transform)

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=pars.batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    encoder, padaa = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    padaa = padaa.to(device)
    encoder.eval()

    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)

    sakt_layer = SAKTLayer(base=64).to(device)
    SC_loss = SCLoss()
    optimizer_proj = torch.optim.Adam(list(sakt_layer.parameters()), lr=pars.proj_lr, betas=(0.5, 0.999))
    optimizer_distill = torch.optim.Adam(list(decoder.parameters()) + list(padaa.parameters()), lr=pars.distill_lr,
                                         betas=(0.5, 0.999))

    best_score = 0
    best_epoch = 0
    best_auroc_px = 0
    best_auroc_sp = 0
    best_aupro_px = 0

    auroc_px_list = []
    auroc_sp_list = []
    aupro_px_list = []

    loss_proj = []
    loss_distill = []
    total_loss = []

    history_infor = {}

    # set appropriate epochs for specific classes (Some classes converge faster than others)
    if _class_ in ['carpet']:
        num_epoch = 200

    print(f'with class {_class_}, Training with {num_epoch} Epoch')

    for epoch in tqdm(range(1, num_epoch + 1)): 
        padaa.train()
        sakt_layer.train()
        decoder.train()
        loss_proj_running = 0
        loss_distill_running = 0
        total_loss_running = 0

        ## gradient acc
        accumulation_steps = 2

        for i, (img, img_noise, _) in enumerate(train_dataloader):
            img = img.to(device)
            img_noise = img_noise.to(device)
            inputs = encoder(img)
            inputs_noise = encoder(img_noise)


            (feature_space_noise, feature_space) = sakt_layer(inputs, features_noise=inputs_noise)


            L_proj = SC_loss(feature_space_noise, feature_space)

            outputs = decoder(padaa(feature_space))  # bn(inputs))

            L_distill = loss_fucntion(inputs, outputs)

            loss = L_distill + pars.weight_proj * L_proj
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                max_grad_norm = 5.0
                torch.nn.utils.clip_grad_norm_(sakt_layer.parameters(), max_grad_norm)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_grad_norm)
                torch.nn.utils.clip_grad_norm_(padaa.parameters(), max_grad_norm)

                optimizer_proj.step()
                optimizer_distill.step()

                optimizer_proj.zero_grad()
                optimizer_distill.zero_grad()


            total_loss_running += loss.detach().cpu().item()
            loss_proj_running += L_proj.detach().cpu().item()
            loss_distill_running += L_distill.detach().cpu().item()


        auroc_px, auroc_sp, aupro_px = evaluation_multi_proj(encoder, sakt_layer, padaa, decoder, test_dataloader, device)
        auroc_px_list.append(auroc_px)
        auroc_sp_list.append(auroc_sp)
        aupro_px_list.append(aupro_px)
        loss_proj.append(loss_proj_running)
        loss_distill.append(loss_distill_running)
        total_loss.append(total_loss_running)


        figure = plt.gcf()  # get current figure
        figure.set_size_inches(8, 12)
        fig, ax = plt.subplots(3, 2, figsize=(8, 12))
        ax[0][0].plot(auroc_px_list)
        ax[0][0].set_title('auroc_px')
        ax[0][1].plot(auroc_sp_list)
        ax[0][1].set_title('auroc_sp')
        ax[1][0].plot(aupro_px_list)
        ax[1][0].set_title('aupro_px')
        ax[1][1].plot(loss_proj)
        ax[1][1].set_title('loss_proj')
        ax[2][0].plot(loss_distill)
        ax[2][0].set_title('loss_distill')
        ax[2][1].plot(total_loss)
        ax[2][1].set_title('total_loss')
        plt.savefig(pars.save_folder + '/' + _class_ + '/monitor_traning.png', dpi=100)
        plt.close('all')

        print(
            'Epoch {}, Sample Auroc: {:.4f}, Pixel Auroc:{:.4f}, Pixel Aupro: {:.4f}'.format(epoch, auroc_sp, auroc_px,
                                                                                             aupro_px))

        if (auroc_px + auroc_sp + aupro_px) / 3 > best_score:
            best_score = (auroc_px + auroc_sp + aupro_px) / 3

            best_auroc_px = auroc_px
            best_auroc_sp = auroc_sp
            best_aupro_px = aupro_px
            best_epoch = epoch


            torch.save({'proj': sakt_layer.state_dict(),
                        'decoder': decoder.state_dict(),
                        'bn': padaa.state_dict()}, save_model_path)


            history_infor['auroc_sp'] = best_auroc_sp
            history_infor['auroc_px'] = best_auroc_px
            history_infor['aupro_px'] = best_aupro_px
            history_infor['epoch'] = best_epoch


            with open(os.path.join(pars.save_folder + '/' + _class_, f'history.json'), 'w') as f:
                json.dump(history_infor, f)
    return best_auroc_sp, best_auroc_px, best_aupro_px


if __name__ == '__main__':
    pars = get_args()
    print('Training with classes: ', pars.classes)
    all_classes = ['carpet','grid','leather','tile', 'wood', 'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
                   'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
    setup_seed(111)
    metrics = {'class': [], 'AUROC_sample': [], 'AUROC_pixel': [], 'AUPRO_pixel': []}

    # train all_classes
    # for c in all_classes:
    for c in pars.classes:
        auroc_sp, auroc_px, aupro_px = train(c, pars)
        print(
            'Best score of class: {}, Auroc sample: {:.4f}, Auroc pixel:{:.4f}, Pixel Aupro: {:.4f}'.format(c, auroc_sp,
                                                                                                            auroc_px,
                                                                                                            aupro_px))
        metrics['class'].append(c)
        metrics['AUROC_sample'].append(auroc_sp)
        metrics['AUROC_pixel'].append(auroc_px)
        metrics['AUPRO_pixel'].append(aupro_px)
        pd.DataFrame(metrics).to_csv(f'{pars.save_folder}/metrics_results.csv', index=False)
