import numpy as np
from init_point_map import sort_point
from PIL import Image
import multiprocessing
import torch


def update_point(point, mask, pred_mask_onehot, img_name, class_num=15, radius=2):
    point_r = np.zeros_like(point)
    pred_mask = np.argmax(pred_mask_onehot, -1)
    pred_mask[np.where(pred_mask == 0)] = class_num
    weight_mask = np.load(img_name.replace("image", "train_point_r_weight").replace('bmp', 'npy'))

    for n in range(1, class_num):
        # p_loc is the key_point local set of the nth class
        p_loc = np.argwhere(point == n)
        key_point = list(p_loc)
        if len(key_point) == 0:
            continue
        # bound is the board loacl set of the nth class
        bound = list(np.argwhere(point == (n + class_num + 1)))

        # increase weight for wrong keypoint
        for i in range(len(key_point)):
            x = key_point[i][0]
            y = key_point[i][1]
            if mask[x, y] != pred_mask[x, y]:
                conf = pred_mask_onehot[x, y, n]
                assert conf < 0.5
                weight_mask[x, y] +=  (0.5 - conf) * 2
            else:
                weight_mask[x, y] = 1

        # find the wrong boardpoint in bound
        p_loc_w = {}
        for i in range(len(bound)):
            x = bound[i][0]
            y = bound[i][1]
            if mask[x, y] != pred_mask[x, y]:
                conf = pred_mask_onehot[x, y, n]
                assert conf < 0.5
                p_loc_w[conf] = bound[i]
        # sort wrong boardpoint with conf
        sorted(p_loc_w)
        # increase weight for wrong boardpoint
        add_point_num = min(len(p_loc_w), len(p_loc_w))
        count = 0
        for conf, p in p_loc_w.items():
            if weight_mask[p[0], p[1]] == 1:
                weight_mask[p[0], p[1]] = 2.
            weight_mask[p[0], p[1]] += (0.5 - conf) * 2
            key_point.append(p)
            count += 1
            if count >= add_point_num:
                break

        key_point = np.array(key_point)
        center = np.mean(key_point, axis=0)
        p_loc = sort_point(key_point, center)
        for kp in p_loc:
            point[kp[0]][kp[1]] = n
            x = kp[0]
            y = kp[1]
            weight_value = weight_mask[x, y]
            s1 = max(0, x - radius)
            e1 = min(x + radius, point_r.shape[0])
            s2 = max(0, y - radius)
            e2 = min(y + radius, point_r.shape[1])
            for x1 in range(s1, e1):
                for y1 in range(s2, e2):
                    if (x1 - x) ** 2 + (y1 - y) ** 2 < radius ** 2 and mask[x1][y1] == n:
                        point_r[x1, y1] = n
                        weight_mask[x1, y1] = weight_value

    point = Image.fromarray(np.uint8(point))
    save_path = img_name.replace("image", "train_point_mask")
    point.save(save_path)

    point_r = Image.fromarray(np.uint8(point_r))
    save_path_r = img_name.replace("image", "train_point_r_mask")
    point_r.save(save_path_r)

    save_path_r_weight = img_name.replace("image", "train_point_r_weight").replace('bmp', 'npy')

    np.save(save_path_r_weight, weight_mask)


def update_point_maps(model, imgs, points, masks, img_names, class_num=15, radius=3):
    points_list = []
    masks_list = []
    pred_masks_list = []
    img_names_list = []
    pred_pros = model(imgs)[-1]

    points = points.numpy()
    pred_pros = torch.nn.functional.softmax(pred_pros, dim=1)
    pred_pros = pred_pros.permute(0, 2, 3, 1).cpu().numpy()
    masks = masks.numpy()

    points_list.extend(points)
    masks_list.extend(masks)
    pred_masks_list.extend(pred_pros)
    img_names_list.extend(img_names)

    pool = multiprocessing.Pool(processes = 40)
    for m in range(len(points_list)):
        pool.apply_async(update_point, (points_list[m], masks_list[m], pred_masks_list[m], img_names_list[m], class_num, radius))
    pool.close()
    pool.join()



