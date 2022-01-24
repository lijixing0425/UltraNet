import matplotlib.pyplot as plt
import numpy as np
import json
from PIL import Image

with open('train_data.json', 'r') as f:
    train_data_path = json.load(f)

plt.figure()
for img_path in train_data_path:
    # if img_path.split('/')[-1] == '1040.bmp':
    img = Image.open(img_path).convert('L')
    mask = Image.open(img_path.replace('image', 'label')).convert('L')
    point_mask = Image.open(img_path.replace('image', 'train_point_mask')).convert('L')
    point_r_mask = Image.open(img_path.replace('image', 'train_point_r_mask')).convert('L')
    point_r_weight = np.load(img_path.replace("image", "train_point_r_weight").replace('bmp', 'npy'))
    # plt.subplot(1, 5, 1)
    # plt.imshow(np.array(img), cmap='viridis')
    plt.subplot(1, 4, 1)
    plt.imshow(np.array(mask), cmap='viridis')
    plt.subplot(1, 4, 2)
    plt.imshow(np.array(point_mask), cmap='viridis')
    plt.subplot(1, 4, 3)
    plt.imshow(np.array(point_r_mask), cmap='viridis')
    plt.subplot(1, 4, 4)
    plt.imshow(np.array(point_r_weight), cmap='viridis')
    print(np.array(point_r_weight).shape)
    print(np.min(point_r_weight), np.max(point_r_weight))
    print(np.min(img), np.max(img))
    print(np.min(mask), np.max(mask))
    print(np.min(point_mask), np.max(point_mask))
    print(np.min(point_r_mask), np.max(point_r_mask))
    plt.show()

