import os
import random

train_path = 'train_classification.txt'
val_path = 'val_classification.txt'
image_path = r'F:\项目\ldh\my_CAM\city\images'

with open(train_path, 'w') as f_train, open(val_path, 'w') as f_val:
    items = os.listdir(image_path)
    random.shuffle(items)

    train_items = items[:-500]
    val_items = items[-500:]

    for item in train_items:
        content = item + '\n'
        f_train.write(content)

    for item in val_items:
        content = item + '\n'
        f_val.write(content)
