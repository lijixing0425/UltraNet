from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch


def saveResult_thyroid(img_path, pred_mask, show=False):

    mask_path = img_path.replace('image', 'label')
    img = np.array(Image.open(img_path).convert('L'))
    mask = np.array(Image.open(mask_path).convert('L'))
    # pred_mask[np.where(pred_mask == 0)] = 4

    plt.figure()
    # plt.subplot(1, 3, 1)
    # plt.imshow(img, cmap='viridis')
    # plt.subplot(1, 3, 2)
    # plt.imshow(mask, cmap='viridis')
    # plt.subplot(1, 3, 3)
    plt.imshow(pred_mask, cmap='viridis')
    # if show:
    #     plt.show()
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig('./result_thyroid/' + mask_path.split('/')[-1].replace('bmp', 'png'), pad_inches=0)
    plt.close()

def saveResult_supersound(img_path, pred_mask, show=False):

    mask_path = img_path.replace('image', 'label')
    img = np.array(Image.open(img_path).convert('L'))
    mask = np.array(Image.open(mask_path).convert('L'))
    # pred_mask[np.where(pred_mask == 0)] = 15

    plt.figure()
    # plt.subplot(1, 3, 1)
    # plt.imshow(img, cmap='viridis')
    # plt.subplot(1, 3, 2)
    # plt.imshow(mask, cmap='viridis')
    # plt.subplot(1, 3, 3)
    plt.imshow(pred_mask, cmap='viridis')
    # if show:
    #     plt.show()
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig('./result_supersound/' + mask_path.split('/')[-1].replace('bmp', 'png'), pad_inches=0)
    plt.close()

def cal_supersound(img_path, pred_mask):

    class_iou = [0.] * 15
    class_pa = [0.] * 15

    mask_path = img_path.replace('image', 'label')
    mask = np.array(Image.open(mask_path).convert('L'))
    pred_mask[np.where(pred_mask == 0)] = 15
    pa = np.sum(mask == pred_mask) / (np.sum(mask == pred_mask) + np.sum(mask != pred_mask))

    mask[np.where(mask == 15)] = 0
    pred_mask[np.where(pred_mask == 15)] = 0
    mask_onehot = torch.nn.functional.one_hot(torch.from_numpy(mask).long(), 15).numpy()
    pred_mask_onehot = torch.nn.functional.one_hot(torch.from_numpy(pred_mask).long(), 15).numpy()
    assert mask_onehot.shape == pred_mask_onehot.shape
    for class_id in range(15):
        mask_onehot_class = mask_onehot[..., class_id]
        pred_mask_onehot_class = pred_mask_onehot[..., class_id]
        class_iou[class_id] = (0.1 + np.sum(mask_onehot_class * pred_mask_onehot_class)) / (0.1 + np.sum(mask_onehot_class) + np.sum(pred_mask_onehot_class) - np.sum(mask_onehot_class * pred_mask_onehot_class))
        class_pa[class_id] = (0.1 + np.sum(mask_onehot_class * pred_mask_onehot_class)) / (0.1 + np.sum(mask_onehot_class))
    class_iou[0], class_iou[-1] = class_iou[-1], class_iou[0]
    class_pa[0], class_pa[-1] = class_pa[-1], class_pa[0]

    return pa, np.array(class_pa), np.array(class_iou)

def cal_thyroid(img_path, pred_mask):

    class_iou = [0.] * 4
    class_pa = [0.] * 4

    mask_path = img_path.replace('image', 'label')
    mask = np.array(Image.open(mask_path).convert('L'))
    pa = np.sum(mask == pred_mask) / (np.sum(mask == pred_mask) + np.sum(mask != pred_mask))
    mask_onehot = torch.nn.functional.one_hot(torch.from_numpy(mask).long(), 15).numpy()
    pred_mask_onehot = torch.nn.functional.one_hot(torch.from_numpy(pred_mask).long(), 15).numpy()
    assert mask_onehot.shape == pred_mask_onehot.shape
    for class_id in range(4):
        mask_onehot_class = mask_onehot[..., class_id]
        pred_mask_onehot_class = pred_mask_onehot[..., class_id]
        class_iou[class_id] = (0.1 + np.sum(mask_onehot_class * pred_mask_onehot_class)) / (0.1 + np.sum(mask_onehot_class) + np.sum(pred_mask_onehot_class) - np.sum(mask_onehot_class * pred_mask_onehot_class))
        class_pa[class_id] = (0.1 + np.sum(mask_onehot_class * pred_mask_onehot_class)) / (0.1 + np.sum(mask_onehot_class))
    return pa, np.array(class_pa), np.array(class_iou)


