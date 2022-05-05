from ast import arg
import os
from src.siamese import *
import torch.utils.data as data
from PIL import Image, ImageOps
import pickle
import numpy as np
from torchvision import transforms
import json

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch import nn
import sys
from collections import OrderedDict

from src.adv_attack.attack.Attack import *

# os.environ["CUDA_VISIBLE_DEVICES"]="1"

# data_folder = './datasets/Sampled_phish1000/'
# annot_file = './datasets/phish1000_coord.txt'
# os.makedirs('./datasets/Sample_phish1000_crop', exist_ok=True)
# os.makedirs(data_folder)

# for brand in os.listdir(data_folder):
#     img = Image.open(data_folder + brand + '/shot.png')

#     ## get ground-truth 
#     with open(annot_file, 'r') as annotation_file:
#         for num, line in enumerate(annotation_file):
#             annotation = line.strip().split(',')
#             site = ','.join(annotation[:-4])
#             if site == brand:
#                 bbox_data_gt = np.array(list(annotation[-4:]))
#                 if len(bbox_data_gt) != 0:
#                     bboxes_gt = bbox_data_gt[:4]
#                     x_min_gt, y_min_gt, x_max_gt, y_max_gt = list(map(float, bboxes_gt))
#                     gt_bbox = [x_min_gt, y_min_gt, x_max_gt, y_max_gt]
#                     break   
#     # print(gt_bbox)
#     cropped = img.crop((x_min_gt, y_min_gt, x_max_gt, y_max_gt))
#     cropped.save(os.path.join('./datasets/Sample_phish1000_crop', brand+'.png'))
#     del gt_bbox


# data_folder = './datasets/Sample_benign1000/'
# annot_file = './datasets/benign1000_coord.txt'
# os.makedirs('./datasets/Sample_benign1000_crop', exist_ok=True)

# for brand in os.listdir(data_folder):
#     img = Image.open(data_folder + brand + '/shot.png')

#     ## get ground-truth 
#     with open(annot_file, 'r') as annotation_file:
#         for num, line in enumerate(annotation_file):
#             annotation = line.strip().split(',')
#             site = ','.join(annotation[:-4])
#             if site == brand:
#                 bbox_data_gt = np.array(list(annotation[-4:]))
#                 if len(bbox_data_gt) != 0:
#                     bboxes_gt = bbox_data_gt[:4]
#                     x_min_gt, y_min_gt, x_max_gt, y_max_gt = list(map(float, bboxes_gt))
#                     gt_bbox = [x_min_gt, y_min_gt, x_max_gt, y_max_gt]
#                     break   
# #     print(gt_bbox)
#     cropped = img.crop((x_min_gt, y_min_gt, x_max_gt, y_max_gt))
#     cropped.save(os.path.join('./datasets/Sample_benign1000_crop', brand+'.png'))
#     del gt_bbox

class GetLoader(data.Dataset):
    def __init__(self, data_root, label_dict, transform=None, grayscale=False):
        
        self.transform = transform
        self.data_root = data_root
        self.grayscale = grayscale

        with open(label_dict, 'rb') as handle:
            self.label_dict = pickle.load(handle)

        self.classes = list(self.label_dict.keys())
        # print(len(self.classes))

        self.n_data = len(os.listdir(self.data_root))

        self.img_paths = []
        self.labels = []

        for data in os.listdir(self.data_root):
            save_path = data
            image_path = data
            label = image_path.split('+')[0]
            
            # deal with inconsistency in naming 
            if brand_converter(label) == 'Microsoft':
                self.labels.append(label)
                
            elif brand_converter(label) == 'DHL Airways':
                self.labels.append('DHL')
                
            elif brand_converter(label) == 'DGI French Tax Authority':
                self.labels.append('DGI (French Tax Authority)')
                
            else:
                self.labels.append(brand_converter(label))

            self.img_paths.append(image_path)

    def __getitem__(self, item):

        img_path, label= self.img_paths[item], self.labels[item]
        img_path_full = os.path.join(self.data_root, img_path)
        # print(img_path)
        
        if self.grayscale:
            img = Image.open(img_path_full).convert('L').convert('RGB')
        else:
            img = Image.open(img_path_full).convert('RGB')

        img = ImageOps.expand(img, (
            (max(img.size) - img.size[0]) // 2, (max(img.size) - img.size[1]) // 2,
            (max(img.size) - img.size[0]) // 2, (max(img.size) - img.size[1]) // 2), fill=(255, 255, 255))
        
        label = self.label_dict[label]
        if self.transform is not None:
            img = self.transform(img)
            
        return img, label, img_path

    def __len__(self):
        return self.n_data


# valid_set = GetLoader(data_root='./adv_attacks/adversarial_relu_jsma/', label_dict='./datasets/target_dict.json', transform=img_transforms)
# valid_set = GetLoader(data_root='./compressed_images/factor_6/', label_dict='./datasets/target_dict.json', transform=img_transforms)


with open('./datasets/target_dict.json', 'rb') as handle:
    label_dict = pickle.load(handle)
    
# valid_loader = torch.utils.data.DataLoader(
#   valid_set, batch_size=1, shuffle=False, pin_memory=True, drop_last=False)


# print(len(valid_loader))

def compute_acc(dataloader, model, device):
    correct = 0
    total = 0

    for b, (x, y, z) in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            pred_cls = torch.argmax(logits, dim=1)

            if not (torch.eq(pred_cls, y).item()):
                print(z)

            correct += torch.sum(torch.eq(pred_cls, y)).item()
            # print('\n')
            # print(pred_cls)
            # print(y)
            total += y.shape[0]
            
    print("Correct: " + str(correct))        
    print('Accuracy: {}'.format(correct/total))    
    return correct/total

class QuantizeRelu(nn.Module):
    def __init__(self, step_size = 0.01):
        super().__init__()
        self.step_size = step_size

    def forward(self, x):
        mask = torch.ge(x, 0).bool() # mask for positive values
        quantize = torch.ones_like(x) * self.step_size
        out = torch.mul(torch.floor(torch.div(x, quantize)), self.step_size) # quantize by step_size
        out = torch.mul(out, mask) # zero-out negative values
        out = torch.abs(out) # remove sign
        return out

def load_model(model_type):
    # Initialize model
    model = KNOWN_MODELS["BiT-M-R50x1"](head_size=277, zero_head=True)

    # Load weights
    checkpoint = torch.load('./src/siamese_pedia/resnetv2_rgb_new.pth.tar', 
                            map_location="cpu")["model"]

    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)

    if (model_type == 'steprelu'):
        print('Testing step relu model')
        model.body.block4.unit01.relu = QuantizeRelu()
        model.body.block4.unit02.relu = QuantizeRelu()
        model.body.block4.unit03.relu = QuantizeRelu()
    else:
        print('Testing relu model')

    model.to(device)
    model.eval()

    return model

if __name__ == "__main__":
    args = sys.argv

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    img_transforms = transforms.Compose(
        [transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args[1])
    # valid_set = GetLoader(data_root='./datasets/Sample_phish1000_crop/', 
    #                 label_dict='./datasets/target_dict.json',
    #                 transform=img_transforms)
    # ['WhatsApp+2020-05-29-13`40`59.png']
    valid_set = GetLoader(data_root='./compressed_images/png_compress_20', 
                    label_dict='./datasets/target_dict.json',
                    transform=img_transforms)
    valid_loader = torch.utils.data.DataLoader(
                valid_set, batch_size=1, shuffle=False, pin_memory=True, drop_last=False)

    if (args[2] == '-accuracy'):
        compute_acc(valid_loader, model, device)
    elif (args[2] in ['fgsm', 'jsma', 'stepll', 'deepfool', 'square']):
        if (len(args) > 3):
            print('Saving successfull attacks')
            check = adversarial_attack(method=args[2], model=model, dataloader=valid_loader, 
                            device=device, num_classes=277, save_data=True)
        else:
            check = adversarial_attack(method=args[2], model=model, dataloader=valid_loader, 
                            device=device, num_classes=277, save_data=False)
        acc, _ = check.batch_attack(args[1])
        print('Model accuracy under ' + args[2] + ' attack: ' + str(acc))
