import multiprocessing
import math
import numpy as np
import cv2
import random
from PIL import Image
import json


def sort_point(points, center):
    tan_list = []
    for point in points:
        tan = math.atan2(point[1] - center[1], point[0] - center[0])
        tan_list.append(tan)
    tan_list = np.asarray(tan_list)
    sort_ind = np.argsort(tan_list)
    points = points[sort_ind]
    return points


def get_boundary(img, point_num):
    max_iou = 0
    max_boundary = []
    img = np.asarray(img, np.uint8)

    pix_sum = np.sum(img)
    if pix_sum < 20:
        return [], []

    boundary_r, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(boundary_r) > 1:
        mx_index = 0
        mx_len = 0
        for idx, b in enumerate(boundary_r):
            if len(b) > mx_len:
                mx_len = len(b)
                mx_index = idx
        boundary = boundary_r[mx_index]
    elif len(boundary_r) == 1:
        boundary = boundary_r[0]
    boundary = np.squeeze(boundary)
    if len(boundary) < 4:
        return [], []
    elif len(boundary) < point_num:
        return boundary, boundary
    for i in range(100):
        ranumber = random.sample(range(0, len(boundary)), point_num)
        center = np.mean(boundary, axis=0)
        extract_boundary = boundary[ranumber]
        extract_boundary = sort_point(extract_boundary, center)
        x, y = img.shape
        img_temp = np.zeros((x, y), np.uint8)
        cv2.fillPoly(img_temp, [extract_boundary], (1))
        img_temp = img_temp + img
        inter = np.sum(img_temp[img_temp == 2]) / 2
        union = np.sum(img_temp[img_temp == 1]) + inter
        iou = inter / union
        if iou > max_iou:
            max_iou = iou
            max_boundary = extract_boundary
            if max_iou > 0.80:
                break

    return max_boundary, boundary


def gen_point_map(img_path, num_class, point_num, r):
    # read data
    mask_path = img_path.replace("image", "label")
    mask = Image.open(mask_path)
    mask = np.asarray(mask)

    # onehot code
    new_mask = np.zeros(mask.shape + (num_class,))
    for i in range(1, num_class):
        new_mask[mask == i, i] = 1

    # get point_map
    point_map = np.zeros_like(new_mask[:, :, 0])
    point_map_r = np.zeros_like(new_mask[:, :, 0])
    point_map_r_weight = np.ones_like(new_mask[:, :, 0])
    for j in range(1, new_mask.shape[2]):
        max_boundary, boundary = get_boundary(new_mask[:, :, j], point_num)
        if len(boundary) > 0:
            cv2.polylines(point_map, [boundary], True, j + num_class + 1, 1)

        for idx in range(len(max_boundary)):
            point_map[max_boundary[idx][1],
                      max_boundary[idx][0]] = j
            x = max_boundary[idx][1]
            y = max_boundary[idx][0]
            s1 = max(0, x - r)
            e1 = min(x + r, point_map_r.shape[0])
            s2 = max(0, y - r)
            e2 = min(y + r, point_map_r.shape[1])
            for x1 in range(s1, e1):
                for y1 in range(s2, e2):
                    if (x1 - x) ** 2 + (y1 - y) ** 2 < r ** 2 and new_mask[x1, y1, j] == 1:
                        point_map_r[x1, y1] = j
                        point_map_r_weight[x1, y1] = 2

    save_path = mask_path.replace('label', 'train_point_mask')
    point_map = Image.fromarray(np.uint8(point_map))
    point_map.save(save_path)

    save_path_r = mask_path.replace('label', 'train_point_r_mask')
    point_map = Image.fromarray(np.uint8(point_map_r))
    point_map.save(save_path_r)

    save_path_r_weight = mask_path.replace('label', 'train_point_r_weight')
    np.save(save_path_r_weight.replace('.bmp', ''), point_map_r_weight)


def gen_point_maps(img_path_list, num_class=15, point_num=10, r=5):
    print("generating point map ...")
    pool = multiprocessing.Pool(processes=40)
    for idx, img_path in enumerate(img_path_list):
        pool.apply_async(gen_point_map, (img_path, num_class, point_num, r))
    pool.close()
    pool.join()
    print("finish!")


if __name__ == '__main__':
    with open('train_data_thyroid.json', 'r') as f:
        train_data_path = json.load(f)

    gen_point_maps(train_data_path, 4)