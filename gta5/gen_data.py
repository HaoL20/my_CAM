#!usr/bin/env python
# -*- coding:utf-8 _*-
import os
import torch
import random
import torchvision
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms

from tqdm import tqdm
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

label_colours_19 = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
    (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
    (0, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100),
    (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 0)]


class City_Dataset(data.Dataset):
    def __init__(self, data_root, size=None, num_classes=19, ignore_label=-1, resize=True):

        self.size = size  # resize大小
        self.resize = resize  # 是否resize
        self.data_root = data_root  # 数据集图像路径，eg: home/haol/data/Dataset/cityscape/gtFine
        self.num_classes = num_classes  # 类别数，GTA5-to-Cityscapes为19，SYNTHIA-to-Cityscapes为16
        self.ignore_label = ignore_label  # 忽略的标签序号

        # GTA5-to-Cityscapes 实验中，只考虑共享的19类
        self.id_to_train_id = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                               19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                               26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        # self.need_class = [0, 2, 8, 10]
        # 获取数据集路径列表
        self.items = [i for i in open('day_list.txt')]
        random.seed(1)
        random.shuffle(self.items)
        self.items = self.items[:3000]
        
    def __getitem__(self, index):
        id_image, id_label = self.items[index].strip('\n').split('\t')
        image_path = self.data_root + id_image
        label_path = self.data_root + id_label
        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path)

        if self.resize:
            image = image.resize(self.size, Image.BICUBIC)
            label = label.resize(self.size, Image.NEAREST)

        image = transforms.functional.to_tensor(image)

        # label映射、转tensor
        label = np.asarray(label, np.float32)
        label = self._label_mapping(label)
        label = torch.from_numpy(label.copy())
        # file name
        image_name = id_image.split('/')[-1][:-4]

        label_name = id_label.split('/')[-1][:-4]
        return image, label, image_name, label_name

    def __len__(self):
        return len(self.items)

    def _label_mapping(self, label):
        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_train_id.items():  # 转换为train_id
            label_copy[label == k] = v
        return label_copy


def colorize_mask(mask):  # mask的尺寸为（H，W），保存为P模式的Image

    # 设置调色板
    label_colours = label_colours_19
    palettes = []
    for label_colour in label_colours:
        palettes = palettes + list(label_colour)
    palettes = palettes + [255, 255, 255] * (256 - len(palettes))

    # mask转图片
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')  # 转换为P模式
    new_mask.putpalette(palettes)
    return new_mask



def main():
    data_root = r'F:\dataset\GTA5_1280_720'
    image_path = 'images'
    label_path = 'labels'
    if os.path.exists(image_path):
        print(image_path + "exists!")
        return
    else:
        os.makedirs(image_path)
        os.makedirs(label_path)

    need_class = {'road': 0, 'building': 2, 'vegetation': 8, 'sky': 10}
    class_to_id = {'road': 0, 'building': 1, 'vegetation': 2, 'sky': 3}
    h, w = (1280, 720)
    crop_size = 400
    random_crop = False          # 是按顺序裁剪还是随机裁剪
    random_crop_num = 3         # 随机裁剪的数量
    random_crop_pos = (h - crop_size) // random_crop_num    # 随机裁剪 h 方向的基准坐标

    areas = []

    if not random_crop:
        num_w = int(w / crop_size) + 1
        num_h = int(h / crop_size) + 1
        for i in range(num_w):
            for j in range(num_h):
                w_start, h_start = i * crop_size, j * crop_size
                if i == num_w - 1:
                    w_start = w - crop_size
                if j == num_h - 1:
                    h_start = h - crop_size
                areas.append((w_start, h_start))

    data_set = City_Dataset(data_root, (h, w), resize=False)
    data_loader = data.DataLoader(dataset=data_set, batch_size=1, pin_memory=True, num_workers=2, shuffle=False)

    for i, (images, labels, image_name, label_name) in tqdm(enumerate(data_loader)):
        if random_crop:
            areas = []
            for j in range(random_crop_num):
                h_start = random.randint(random_crop_pos * j, random_crop_pos * (j + 1))
                w_start = random.randint(0, 100)            # 保证多裁剪点天空
                areas.append((w_start, h_start))

        for idx_area, area in enumerate(areas):                             # 遍历全部左上角起点位置
            w_start, h_start = area
            crop_images = images[:, :, w_start:w_start + crop_size, h_start:h_start + crop_size]
            crop_labels = labels[:, w_start:w_start + crop_size, h_start:h_start + crop_size]
            idx, count = torch.unique(crop_labels, return_counts=True)      # 统计每个标签的数量
            cur_label = []                  # 当前裁剪的标签
            is_save = True                  # 是否保存当前裁剪区域，如果   need_class 在里面，但是有某个类别的占比低于阈值，则舍弃不保存！
            for name, label_idx in need_class.items():  # {'road': 0, 'building': 2, 'vegetation': 8, 'sky': 10}找是否存在相应的像素点，并且占比超过20%
                mask = (idx == label_idx)
                if True not in mask:        # need_class不在里面
                    continue
                cur_count = count[mask]     # 当前类别的数量
                present = cur_count / (crop_size ** 2)  # 当前类别占比
                if name == 'sky':           # 天气区域阈值5%
                    threshold = 0.05
                else:
                    threshold = 0.1         # 其他区域阈值10%

                if present > threshold:
                    # cur_label.append("{}:{:.2f}".format(class_to_id[name], present.item() * 100))
                    cur_label.append("{}".format(class_to_id[name]))
                else:
                    is_save = False         # 某个类别的占比低于阈值，则舍弃不保存！
                    break

            if cur_label and is_save:
                img = torchvision.utils.make_grid(crop_images).numpy()  # 使用make_grid的normalize
                img = np.transpose(img, (1, 2, 0)) * 255
                img = Image.fromarray(np.uint8(img))
                img.save('{}/random_{}_{}_{}_class_{}.png'.format(image_path, random_crop, image_name[0], idx_area, ''.join(cur_label)))

                crop_labels = torch.unsqueeze(crop_labels, dim=1)               # (b,h,w)   ==> (b,1,h,w)，make_grid只能处理4维的向量，三维的label必须扩充一个通道的维度
                crop_labels = torchvision.utils.make_grid(crop_labels, nrow=4)  # (b,1,h,w) ==> (3,h,w), 单通道会被扩充到3通道。
                crop_labels = crop_labels.numpy()[0]                            # (3,h,w)   ==> (h,w)，  转换为numpy，取单通道。P模式的图片必须要单通道。

                output_col = colorize_mask(crop_labels)  # 转换为P模式的Image，并且换上对应的调试板，将其可视化。
                output_col.save('{}/random_{}_{}_class_{}.png'.format(label_path, random_crop, label_name[0], idx_area, ''.join(cur_label)))


if __name__ == '__main__':
    main()
