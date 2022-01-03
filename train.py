#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author : LIAOH
@file   : train.py
# @Time : 2022/1/1 22:18
"""

from torch.backends import cudnn

cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F

from net.resnet50_cam import ResNet50_CAM
from misc.utils import *
from dataset.ClassificationDataset import ClassificationDataset


def validate(model, data_loader):
    print('validating ... ', flush=True, end='')
    val_loss_meter = AverageMeter('loss1', 'loss2')

    model.eval()
    with torch.no_grad():
        for pack in data_loader:
            img = pack['img']
            label = pack['label'].cuda(non_blocking=True)
            pred, feat = model(img)
            loss1 = F.multilabel_soft_margin_loss(pred, label)
            val_loss_meter.add({'loss1': loss1.item()})
    model.train()
    print('loss: %.4f' % (val_loss_meter.pop('loss1')))
    return


def run():
    data_path_A = r'F:\project\ldh\my_CAM\city\images'
    data_path_B = r'F:\project\ldh\my_CAM\gta5\images'
    save_model_path = 'sess/res50_cam.pth'
    batch_size = 8
    num_workers = 2
    num_epoches = 20
    learning_rate = 0.001
    weight_decay = 1e-4
    model = ResNet50_CAM()

    train_dataset = ClassificationDataset(data_path_A, data_path_B, split='train')
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)

    val_dataset = ClassificationDataset(data_path_A, data_path_B, split='val')
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)

    max_step = (len(train_dataset) // batch_size) * num_epoches
    param_groups = model.trainable_parameters()
    optimizer = PolyOptimizer([
        {'params': param_groups[0], 'lr': learning_rate, 'weight_decay': weight_decay},
        {'params': param_groups[1], 'lr': 10 * learning_rate, 'weight_decay': weight_decay},
    ], lr=learning_rate, weight_decay=weight_decay, max_step=max_step)

    model = torch.nn.DataParallel(model)
    model.train()

    avg_meter = AverageMeter()

    timer = Timer()

    for ep in range(num_epoches):

        print('Epoch %d/%d' % (ep + 1, num_epoches))
        for step, pack in enumerate(train_data_loader):
            img = pack['img']
            label = pack['label'].cuda(non_blocking=True)

            pred, feat = model(img)
            loss = F.multilabel_soft_margin_loss(pred, label)
            avg_meter.add({'loss1': loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (optimizer.global_step - 1) % 100 == 0:
                timer.update_progress(optimizer.global_step / max_step)
                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f' % (avg_meter.pop('loss1')),
                      'imps:%.1f' % ((step + 1) * batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)

        else:
            validate(model, val_data_loader)
            timer.reset_stage()

    torch.save(model.module.state_dict(), save_model_path)
    torch.cuda.empty_cache()

if __name__ == '__main__':
    run()