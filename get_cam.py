from torchvision import transforms
from torch.autograd import Variable
import numpy as np
import cv2

import torch
from PIL import Image
import net.resnet50_cam


# generate class activation mapping for the top1 prediction
def returnCAM(feature_conv, weight_softmax, class_idx):
    # feature_conv: 特征图
    # weight_softmax： 权重
    # class_idx：可视化的类别

    size_upsample = (300, 300)
    bz, nc, h, w = feature_conv.shape  # 特征图的维度
    output_cam = []  # 保存每个的cam图
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


def get_cam(net, features_blobs, img_pil, classes, root_img, class_idx):
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-1].data.cpu().numpy())

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    logit = net(img_variable)

    CAMs = returnCAM(features_blobs[0], weight_softmax, [class_idx])

    # render the CAM and output
    print('output CAM.jpg for the top1 prediction: %s' % classes[class_idx])
    img = cv2.imread(root_img)
    height, width, _ = img.shape
    CAM = cv2.resize(CAMs[0], (width, height))
    heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    cv2.imwrite('cam_img_{}.jpg'.format(classes[class_idx]), result)


# 加载模型
net = net.resnet50_cam.Net()
net.load_state_dict(torch.load("sess/res50_cam.pth.pth"))


def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())


# hook the feature extractor, 用hook提取stage4后的结果，也就是gap2d之前的结果
features_blobs = []
net._modules.get('stage4').register_forward_hook(hook_feature)

# 类别编号
classes = {0: 'road', 1: 'building', 2: 'vegetat ion', 3: 'sky'}

root = 'berlin_000000_000019_leftImg8bit_road.png'
img = Image.open(root).convert('RGB')
for i in range(4):
    get_cam(net, features_blobs, img, classes, root, i)
