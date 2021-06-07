#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:2021/6/4 9:51
# @Author:Jianyuan Hong
# @File:inference.py
# @Software:Vscode


import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import argparse
import numpy as np
from core.models.model_zoo import get_segmentation_model


parser = argparse.ArgumentParser(
    description='inference segmentation result from a given image or files')
parser.add_argument('--model', type=str, default='unet',
                        choices=['unet',], help='model name (default: fcn32s)') 
parser.add_argument('--dataset', type=str, default='FUSeg',
                        choices=['FUSeg', ],help='dataset name (default: pascal_voc)')
parser.add_argument('--checkpoint', default="/home/hongjianyuan/seg/runs/models/unet_FUSeg_best_model.pth",
                    help='Directory for saving checkpoint models')
parser.add_argument('--input', type=str, default='/home/dataset/Foot_Ulcer Segmentation_Challenge/validation/images/',
                    help='path to the input picture or path to the files')
parser.add_argument('--outdir', default='./eval', type=str,help='path to save the predict result')
parser.add_argument('--devices', type=str, default='0', help='The ids of GPU to be used')
args = parser.parse_args()

def inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # output folder
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # 初始化模型
    model = get_segmentation_model(model=args.model, dataset=args.dataset).to(device)
    model = nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(args.checkpoint))
    print('Finished loading model!')

    # image transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    files = os.listdir(args.input)
    for file in files:
        image_path = os.path.join(args.input,file)
        img = cv2.imread(image_path)
        images = transform(img).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            output = model(images)
            pred = torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy()
            pred = np.where(pred == 1,255, 0)
            cv2.imwrite(args.outdir +'/'+os.path.basename(image_path), pred)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    inference(args)



