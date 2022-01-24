from ce_net import CE_Net_
from dataset import supersound_dataset
from tqdm import tqdm
from loss import MultiClassDiceLoss, point_map_loss
import json
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from init_point_map import gen_point_maps
from update_point_map import update_point_maps
import shutil
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def update(model, train_dataset):
    for img, point, bound_label, label, point_weight, path in tqdm(train_dataset):
        img = img.cuda()
        mask = label
        model.eval()
        with torch.no_grad():
            update_point_maps(model, img, bound_label, mask, path, class_num=15, radius=5)


if __name__ == '__main__':
    # read data
    with open('train_data.json', 'r') as f:
        train_data_path = json.load(f)
    with open('val_data.json', 'r') as f:
        val_data_path = json.load(f)

    data_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = supersound_dataset(train_data_path, data_transforms=data_transform)
    train_dataset = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8, pin_memory=True)

    val_dataset = supersound_dataset(val_data_path, data_transforms=data_transform)
    val_dataset = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True)



    # model setting
    loss_layer = [True, True, False, False,
                   False, False, False, False]
    model = CE_Net_().cuda()

    # model load
    if False:
        model.load_state_dict(torch.load('./record/best_model_5_0.1093.pth'))
        update(model, train_dataset)
        exit()
    else:
        gen_point_maps(train_data_path)
        best_val_loss = 1.
        # best_test_loss = 100.

    from tensorboardX import SummaryWriter
    writer = SummaryWriter()
    x_label = 0
    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(8):
        print('begin the {}th study'.format(epoch))
        end_count = 0

        for i in range(600):
            print('begin the {}th train \n'.format(i))
            train_loss = 0
            train_loss_point = 0
            model.train()
            for img, point, bound_label, label, point_weight, path in tqdm(train_dataset):
                img = img.cuda()
                label = label.cuda()
                point = point.cuda()
                point_weight = point_weight.cuda()
                output = model(img)
                loss = MultiClassDiceLoss(output[-1], label)
                loss_dice = loss.clone()
                for out, loss_flag in zip(output[:-1], loss_layer):
                    if out is not None and loss_flag is True:
                        loss1, loss2 = point_map_loss(out, point, point_weight)
                        loss = loss + loss1*10 + loss2
                train_loss += loss_dice.item()
                train_loss_point += loss2.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            train_loss = train_loss / len(train_dataset)
            train_loss_point = train_loss_point / len(train_dataset)
            print('the {}th train, the mean dice_loss is {} , the point_dice_loss is {},\n'.format(i, train_loss,
                                                                                                   train_loss_point))

            print('start {}th val \n'.format(i))
            val_loss = 0
            model.eval()
            with torch.no_grad():
                for img, label, _ in tqdm(val_dataset):
                    img = img.cuda()
                    label = label.cuda()
                    output = model(img)
                    loss = MultiClassDiceLoss(output[-1], label)
                    val_loss += loss.item()
            val_loss = val_loss / len(val_dataset)
            print('the {}th val, the mean loss is {} \n'.format(i, val_loss))

            writer.add_scalars('./view', {'train_dice_loss': train_loss, 'val_dice_loss': val_loss,
                                          'sub_loss': val_loss - train_loss}, x_label)
            x_label += 1
            if val_loss < best_val_loss:
                end_count = 0
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                end_count += 1
                if end_count > 5:
                    print('end train and the best val_loss is {}'.format(best_val_loss))
                    end_count = 0
                    break


        shutil.copyfile('best_model.pth', './record/best_model_{}_{:.4f}.pth'.format(epoch, best_val_loss))
        model.load_state_dict(torch.load('best_model.pth'))
        update(model, train_dataset)
        best_val_loss = 1

    writer.close()














