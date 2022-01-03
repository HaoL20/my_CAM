import os
import torch
import random
import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ClassificationDataset(Dataset):
    def __init__(self, image_path_A, image_path_B, split='train', random_mirror=True, num_class=4):

        self.random_mirror = random_mirror
        self.num_class = num_class
        list_A = [i for i in open("city/{}_classification.txt".format(split))]
        list_B = [i for i in open("GTA5/{}_classification.txt".format(split))]
        items_A = [os.path.join(image_path_A, i.strip('\n')) for i in list_A]
        items_B = [os.path.join(image_path_B, i.strip('\n')) for i in list_B]
        self.items = items_A + items_B

    def __getitem__(self, index):
        # 读取图片
        name_str = self.items[index]
        image = Image.open(name_str).convert("RGB")

        # 随机翻转
        if self.random_mirror and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_transforms_list = [transforms.ToTensor(),  # (w,h,c) ==> (c,h,w) Converts [0,255] to [0,1]
                                 transforms.Normalize([.485, .456, .406], [.229, .224, .225])]  # ImageNet数据集的均值和方差。因为使用了ImageNet预训练模型，数据应该进行相同的Normalize

        size = (image.size[0], image.size[0])
        # image转tensor
        image_transforms = transforms.Compose(image_transforms_list)
        image = image_transforms(image)

        # 通过图片文件名获取标签
        _, label_str = name_str.split('_class_')        # label_str:'013.png'
        labels = [int(idx) for idx in label_str[:-4]]   # label: [0,1,3]
        labels = torch.tensor(labels, dtype=torch.long) # label to tensor
        yt_onehot = torch.zeros(self.num_class)         # [0,0,0,0]
        yt_onehot.scatter_(0, labels, 1)                # [1,1,0,1]

        return {'name': name_str, 'img': image, 'label': yt_onehot, 'size': size}

    def __len__(self):
        return len(self.items)


def main():
    data_path_A = r'F:\project\ldh\my_CAM\city\images'
    data_path_B = r'F:\project\ldh\my_CAM\gta5\images'

    data_set = ClassificationDataset(data_path_A, data_path_B)
    data_loader = DataLoader(data_set, batch_size=1, shuffle=True)

    for idx, pack in enumerate(data_loader):
        images = pack['img']
        name = pack['name']
        label = pack['label']

        mean = torch.as_tensor([.485, .456, .406], dtype=images.dtype, device=images.device).view(-1, 1, 1)  # (3) ==> (3,1,1)，这样才可以进行后面的广播运算
        std = torch.as_tensor([.229, .224, .225], dtype=images.dtype, device=images.device).view(-1, 1, 1)
        img = images * std + mean

        img = torchvision.utils.make_grid(img, nrow=4).numpy()  # (b,3,h,w) ==> (3,H,W)。其中H，W是把b张图片按照相应规则(每行最多4张图片)拼接成的新图片的尺寸。
        img = np.transpose(img, (1, 2, 0)) * 255  # (3,H,W) ==> (H,W,3)
        # img = img[:, :, ::-1]                                             # 如果加载为BGR的模式，需要转换为RGB的模式
        img = Image.fromarray(np.uint8(img))  # 转换为uint8，再转换为Image
        img.save('Demo_{}.jpg'.format(idx))

        print(name, label)


if __name__ == '__main__':
    os.chdir('..')  # 改变当前工作目录到上一级目录(项目目录)
    main()
