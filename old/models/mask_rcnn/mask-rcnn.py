import random
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import torch.utils.data
import cv2
import torchvision.models.segmentation
import torch
import os
from clevr_dataset import ClevrDataset

batchSize = 2
imageSize = [600, 600]
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def loadData():
    batch_Imgs = []
    batch_Data = []

    # load images and masks
    for i in range(batchSize):
        idx = random.randint(0, len(imgs) - 1)
        img = cv2.imread(os.path.join(imgs[idx], "Image.jpg"))
        img = cv2.resize(img, imageSize, cv2.INTER_LINEAR)
        maskDir = os.path.join(imgs[idx], "Vessels")
        masks = []

        # get bounding box coordinates for each mask
        for mskName in os.listdir(maskDir):
            vesMask = (cv2.imread(maskDir + '/' + mskName, 0) > 0).astype(np.uint8)
            vesMask = cv2.resize(vesMask, imageSize, cv2.INTER_NEAREST)
            masks.append(vesMask)

        num_objs = len(masks)

        # if image have no objects just load another image
        if num_objs == 0: return loadData()

        boxes = torch.zeros([num_objs, 4], dtype=torch.float32)

        for i in range(num_objs):
            x, y, w, h = cv2.boundingRect(masks[i])
            boxes[i] = torch.tensor([x, y, x + w, y + h])

        masks = torch.as_tensor(masks, dtype=torch.uint8)
        img = torch.as_tensor(img, dtype=torch.float32)

        data = {}
        data["boxes"] = boxes
        data["labels"] = torch.ones((num_objs,), dtype=torch.int64)  # there is only one class
        data["masks"] = masks
        batch_Imgs.append(img)
        batch_Data.append(data)

    batch_Imgs = torch.stack([torch.as_tensor(d) for d in batch_Imgs], 0)
    batch_Imgs = batch_Imgs.swapaxes(1, 3).swapaxes(2, 3)

    return batch_Imgs, batch_Data

# load an instance segmentation model pre-trained pre-trained on COCO
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
model.to(device)

optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)
model.train()

for i in range(10001):
    images, targets = loadData()
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    optimizer.zero_grad()
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    losses.backward()
    optimizer.step()

    print(i, 'loss:', losses.item())

    if i % 500 == 0:
        torch.save(model.state_dict(), str(i) + ".torch")

