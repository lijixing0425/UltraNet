from ce_net import CE_Net_
from dataset import supersound_dataset
from tqdm import tqdm
from loss import MultiClassDiceLoss, MultiClassDice, MultiClassiou
import json
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import cal_thyroid, saveResult_supersound, cal_supersound, saveResult_thyroid
import numpy as np
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# read data
with open('val_data.json', 'r') as f:
    test_data_path = json.load(f)
data_transform = transforms.Compose([transforms.ToTensor()])
test_dataset = supersound_dataset(test_data_path, data_transforms=data_transform)
test_dataset = DataLoader(test_dataset, batch_size=1, shuffle=False)

# load model
model = CE_Net_()
model.load_state_dict(torch.load('./record/best_model_0_0.1285.pth'))
model.cuda()

# test model
model.eval()
test_loss = 0
test_dice = 0
test_miou = 0
global_pa = 0.
global_class_pa = np.array([0.] * 15)
global_class_iou = np.array([0.] * 15)

with torch.no_grad():
    for img, label, img_path in tqdm(test_dataset):
        img = img.cuda()
        label = label.cuda()
        output = model(img)
        loss = MultiClassDiceLoss(output[-1], label)
        dice = MultiClassDice(output[-1], label)
        miou = MultiClassiou(output[-1], label)
        test_dice += dice.item()
        test_loss += loss.item()
        test_miou += miou.item()
        output = torch.squeeze(torch.argmax(output[-1], dim=1)).cpu().numpy()
        saveResult_supersound(img_path[0], output, show=False)
        pa, class_pa, class_iou = cal_supersound(img_path[0], output)
        global_pa += pa
        global_class_pa += class_pa
        global_class_iou += class_iou

print('\n')
print('the soft_dice_loss is {} \n'.format(test_loss / len(test_dataset)))
print('the dice is {} \n'.format(test_dice / len(test_dataset)))
print('the miou1 is {} \n'.format(test_miou / len(test_dataset)))

print('the miou2 is {} \n'.format(np.mean(global_class_iou / len(test_dataset))))
print('the class_miou is : \n')
print(pd.DataFrame(global_class_iou / len(test_dataset)))
print('the global_PA is {} \n'.format(global_pa / len(test_dataset)))
print('the mpa is {} \n'.format(np.mean(global_class_pa / len(test_dataset))))
print('the class_pa is : \n')
print(pd.DataFrame(global_class_pa / len(test_dataset)))