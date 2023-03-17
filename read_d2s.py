import json
import os
from pycocotools.coco import COCO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np

json_path = "./dataset/d2s/d2s_annotations_v1.1/annotations/D2S_training.json"
# json_path = "./dataset/d2s/d2s_annotations_v1.1/annotations/D2S_validation_clutter.json"
img_path = "./dataset/d2s/d2s_images_v1/images"

# test_json_path = "./dataset/d2s/d2s_annotations_v1.1/annotations/D2S_validation.json"
# test_json_path = "./dataset/d2s/d2s_annotations_v1.1/annotations/D2S_validation_clutter.json"
test_json_path = "./dataset/d2s/d2s_annotations_v1.1/annotations/D2S_validation_random_background.json"
# json_labels = json.load(open(json_path, 'r'))
# print(json_labels["info"])

# load coco format data
coco = COCO(annotation_file=json_path)
test_coco = COCO(annotation_file=test_json_path)
# get all image index info
ids = list(sorted(coco.imgs.keys()))
print("number of images: {}".format(len(ids)))

# get all class labels
coco_classes = dict([(v['id'], v['name']) for k,v in coco.cats.items()])
print(coco_classes)

first_mask = None
frames = []
filenames = []
masks = []
# show the first three images
avocado = [16100, 16101]
pasta_reggia_spaghetti = [20924, 20925, 20926] #20924-20929, 21000-21029
apple_braeburn_bundle = ids[-8:]
# for img_id in ids[90:92]:
for img_id in pasta_reggia_spaghetti[:2]:
# for img_id in avocado:
    # get annotations idx
    ann_ids = coco.getAnnIds(imgIds=img_id)

    # according to anno idx, get the annos
    targets = coco.loadAnns(ann_ids)
    obj_id = targets[0]['category_id']
    print("obj id: ", obj_id)
    print("found ", coco_classes[obj_id])

    # get image file name
    path = coco.loadImgs(img_id)[0]["file_name"]
    # filenames.append(os.path.join(img_path, path))
    frame = np.array(Image.open(os.path.join(img_path, path)), dtype=float)
    frames.append(frame)
    # print(frame)
    # read image
    img = Image.open(os.path.join(img_path, path)).convert('RGB')
    draw = ImageDraw.Draw(img)
    # draw box to image
    for target in targets:
        x,y,w,h = target['bbox']
        x1, y1, x2, y2 = x, y, int(x+w), int(y+h)
        draw.rectangle((x1,y1,x2,y2))
        draw.text((x1, y1), coco_classes[target["category_id"]])
    fig = plt.figure()
    fig.add_subplot(1,2,1)
    plt.imshow(img)
    plt.axis("off")
    coco.showAnns(targets)

    fig.add_subplot(1,2,2)
    mask = coco.annToMask(targets[0])
    for i in range(1, len(targets)):
        mask += coco.annToMask(targets[i])*(i+1)
    print("image id: ", img_id)
    first_mask = mask[0]
    # if img_id == 200 or img_id == 201 or img_id == 202:
    masks.append(mask)
    # print(mask)
    # print(type(mask))
    # plt.subplots(122)
    plt.imshow(mask)
    plt.axis("off")
    plt.show()


# get test images
test_ids = list(sorted(test_coco.imgs.keys()))
for img_id in test_ids[-5:]:
    # get annotations idx
    ann_ids = test_coco.getAnnIds(imgIds=img_id)
    # according to anno idx, get the annos
    targets = test_coco.loadAnns(ann_ids)
    # if targets[0]['category_id'] == 29:
    #     print("found ", coco_classes[29])
        # get image file name
    path = test_coco.loadImgs(img_id)[0]["file_name"]
    # filenames.append(os.path.join(img_path, path))
    frame = np.array(Image.open(os.path.join(img_path, path)), dtype=float)
    frames.append(frame)
    # print(frame)
    # read image
    img = Image.open(os.path.join(img_path, path)).convert('RGB')
    draw = ImageDraw.Draw(img)
    # draw box to image
    for target in targets:
        x,y,w,h = target['bbox']
        x1, y1, x2, y2 = x, y, int(x+w), int(y+h)
        draw.rectangle((x1,y1,x2,y2))
        draw.text((x1, y1), coco_classes[target["category_id"]])
    fig = plt.figure()
    fig.add_subplot(1,2,1)
    plt.imshow(img)
    plt.axis("off")
    test_coco.showAnns(targets)

    fig.add_subplot(1,2,2)
    mask = test_coco.annToMask(targets[0])
    for i in range(1, len(targets)):
        mask += test_coco.annToMask(targets[i])*(i+1)
    print("image id: ", img_id)
    masks.append(mask)
    # print(mask)
    # print(type(mask))
    # plt.subplots(122)
    plt.imshow(mask)
    plt.axis("off")
    plt.show()

        # break

#

import os
from os import path
from argparse import ArgumentParser
import shutil

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from inference.data.test_datasets import LongTestDataset, DAVISTestDataset, YouTubeVOSTestDataset
from inference.data.mask_mapper import MaskMapper
from model.network import XMem
from inference.inference_core import InferenceCore

from progressbar import progressbar

torch.set_grad_enabled(False)
if torch.cuda.is_available():
  print('Using GPU')
  device = 'cuda'
else:
  print('CUDA not available. Please connect to a GPU instance if possible.')
  device = 'cpu'
# default configuration
config = {
    'top_k': 30,
    'mem_every': 5,
    'deep_update_every': -1,
    'enable_long_term': True,
    'enable_long_term_count_usage': True,
    'num_prototypes': 128,
    'min_mid_term_frames': 5,
    'max_mid_term_frames': 10,
    'max_long_term_elements': 10000,
}

network = XMem(config, './saves/XMem.pth').eval().to(device)
# print(network)

# video_name = 'video.mp4'
# mask_name = 'first_frame.png'


# from base64 import b64encode
# data_url = "data:video/mp4;base64," + b64encode(open(video_name, 'rb').read()).decode()
# import IPython.display
# IPython.display.Image('first_frame.png', width=400)

mask = first_mask #np.array(Image.open(mask_name))
# print(np.unique(mask))
num_objects = 1 #len(np.unique(mask)) - 1

import cv2
from inference.interact.interactive_utils import image_to_torch, index_numpy_to_one_hot_torch, torch_prob_to_numpy_mask, overlay_davis

torch.cuda.empty_cache()

processor = InferenceCore(network, config=config)
processor.set_all_labels(range(1, num_objects+1)) # consecutive labels
# cap = cv2.VideoCapture(video_name)

# You can change these two numbers
# frames_to_propagate = 40
visualize_every = 20

current_frame_index = 0

import matplotlib.pyplot as plt
with torch.cuda.amp.autocast(enabled=True):
    for frame in frames:
        # convert numpy array to pytorch tensor format
        frame_torch, _ = image_to_torch(frame, device=device)
        if current_frame_index <= 1:
        # if current_frame_index == 0:
            # initialize with the mask
            mask_torch = index_numpy_to_one_hot_torch(masks[current_frame_index], num_objects + 1).to(device)
            # mask_torch = index_numpy_to_one_hot_torch(mask, num_objects + 1).to(device)
            # the background mask is not fed into the model
            prediction = processor.step(frame_torch, mask_torch[1:])
        else:
            # propagate only
            prediction = processor.step(frame_torch)

        # argmax, convert to numpy
        prediction = torch_prob_to_numpy_mask(prediction)

        # if current_frame_index % visualize_every == 0:
        print("frame: ", frame.shape)
        visualization = overlay_davis(frame, prediction)

        imgplot = plt.imshow(visualization.astype(np.uint8))
        plt.axis("off")
        plt.show()

        current_frame_index += 1
#
