import json
import os
from pycocotools.coco import COCO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random

# json_path = "./dataset/d2s/d2s_annotations_v1.1/annotations/D2S_training.json"
img_path = "./dataset/d2s/d2s_images_v1/images"

clutter_json_path = "./dataset/d2s/d2s_annotations_v1.1/annotations/D2S_validation_clutter.json"
random_json_path = "./dataset/d2s/d2s_annotations_v1.1/annotations/D2S_validation_random_background.json"
test_json_path = "./dataset/d2s/d2s_annotations_v1.1/annotations/D2S_training.json"
json_path = "./dataset/d2s/d2s_annotations_v1.1/annotations/D2S_training.json"
light_json_path = "./dataset/d2s/d2s_annotations_v1.1/annotations/D2S_training_light0.json"

# json_labels = json.load(open(json_path, 'r'))
# print(json_labels["info"])

# load coco format data
coco = COCO(annotation_file=json_path)
clutter_coco = COCO(annotation_file=clutter_json_path)
random_coco = COCO(annotation_file=random_json_path)
light_coco = COCO(annotation_file=light_json_path)

test_coco = COCO(annotation_file=test_json_path)
# get all image index info
ids = list(sorted(test_coco.imgs.keys()))
print("number of images: {}".format(len(ids)))

# get all class labels
coco_classes = dict([(v['id'], v['name']) for k,v in test_coco.cats.items()])
# print(coco_classes)

first_mask = None
frames = []
filenames = []
masks = []

# one annotation example
print("one annotation example")
for key in coco.anns.keys():
    print(coco.anns[key])
    break

# one category example
print("one category example")
print(coco.cats[10])

# one image example
print("one image example")
for key in coco.imgs.keys():
    print(coco.imgs[key])
    break

# get all coco categories
print("all coco categories")
print(coco.getCatIds()[:5])

# get metadata of coco categories
# print(coco.loadCats(coco.getCatIds())[:5])
# print(coco.loadCats(5))

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
# print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
# print('COCO supercategories: \n{}'.format(' '.join(nms)))

# get all images containing given categories, select one at random
# catIds = coco.getCatIds(catNms=['gepa_bio_und_fair_kraeuterteemischung'])
# print("catIds: {}".format(catIds))
# imgIds = coco.getImgIds(catIds=catIds)
# print("imgIds: {}".format(imgIds))
#
# imgIds = coco.getImgIds(imgIds = [200])
# print("imgIds: {}".format(imgIds))
#
# # get images that have all category ids
# imgIds = coco.getImgIds(catIds=[47, 50, 51])
# print("imgIds: {}".format(imgIds))

# load and display image
def display_image(coco, img_id):
    I = Image.open(os.path.join(img_path, coco.imgs[img_id]['file_name']))
    plt.axis('off')
    plt.imshow(I)
    plt.show()

def imgID_to_image(coco, img_id):
    im = coco.loadImgs(img_id)
    print("im: {}".format(im[0]))

    I = Image.open(os.path.join(img_path, im[0]['file_name']))
    plt.axis('off')
    plt.imshow(I)
    plt.show()

# clutter image
# display COCO categories and supercategories
cats = clutter_coco.loadCats(clutter_coco.getCatIds())
# nms=[cat['name'] for cat in cats]
# print('Clutter COCO categories: \n{}\n'.format(' '.join(nms)))
#
# nms = set([cat['supercategory'] for cat in cats])
# print('Clutter COCO supercategories: \n{}'.format(' '.join(nms)))
#
# my_catIds = clutter_coco.getCatIds(catNms=['gepa_bio_und_fair_kamillentee'])
# my_catIds2 = coco.getCatIds(catNms=['gepa_bio_und_fair_kamillentee'])
# print("my_catIds: {}".format(my_catIds))
# print("my_catIds2: {}".format(my_catIds2))
def cat_to_images(coco_file, cat_ids):
    imgIds = coco_file.getImgIds(catIds=cat_ids)
    # print("imgIds: {}".format(imgIds))
    # print("number of images: {}".format(len(imgIds)))
    return imgIds

def vis_color_and_mask(image, mask):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # Display the first image in the first subplot
    axs[0].imshow(image)
    axs[0].set_title('Image')
    # Display the second image in the second subplot
    axs[1].imshow(mask) # , cmap='gray'
    axs[1].set_title('Its mask')
    plt.show()

def split_image_with_mask(color, mask, ignore=None):
    # plt.imshow(color)
    # plt.show()
    mask_ids = np.sort(np.unique(mask))
    color = np.array(color).astype(int)
    # print("sorted mask ids: ", mask_ids)
    sub_images = []
    sub_masks = []
    for i in mask_ids:
        if i in ignore:
            continue
        temp_mask = (mask==i).astype(int)
        sub_image = color.copy()
        sub_mask = mask.copy()
        # set all pixels not in the mask to 0
        sub_image[temp_mask==0] = 0
        sub_mask[temp_mask==0] = 0
        # set all labels to 1
        sub_mask[sub_mask>0] = 1
        sub_images.append(sub_image)
        sub_masks.append(sub_mask)
        # print("unique values in sub mask: ", np.unique(sub_mask))
        # visualize the image and its mask
    # vis_color_and_mask(sub_images[0], sub_masks[0])
    return sub_images, sub_masks

def imgID_to_sample(coco, img_id):
    ann_ids = coco.getAnnIds(imgIds=img_id)

    # according to anno idx, get the annos
    targets = coco.loadAnns(ann_ids)
    obj_id = targets[0]['category_id']
    # print("obj id: ", obj_id)
    # print("found ", coco_classes[obj_id])

    # get image file name
    path = coco.loadImgs(img_id)[0]["file_name"]
    # filenames.append(os.path.join(img_path, path))
    frame = np.array(Image.open(os.path.join(img_path, path)), dtype=float)
    # frames.append(frame)
    # print(frame)
    # read image
    img = Image.open(os.path.join(img_path, path)).convert('RGB')
    raw_img = img.copy()
    draw = ImageDraw.Draw(img)
    # draw box to image
    for target in targets:
        x, y, w, h = target['bbox']
        x1, y1, x2, y2 = x, y, int(x + w), int(y + h)
        draw.rectangle((x1, y1, x2, y2))
        draw.text((x1, y1), coco_classes[target["category_id"]])
    # fig = plt.figure()
    # fig.add_subplot(1, 2, 1)
    # plt.imshow(img)
    # plt.axis("off")
    coco.showAnns(targets)

    # fig.add_subplot(1, 2, 2)
    mask = coco.annToMask(targets[0])
    for i in range(1, len(targets)):
        mask += coco.annToMask(targets[i]) * (i + 1)
    # print("image id: ", img_id)
    first_mask = mask[0]
    # if img_id == 200 or img_id == 201 or img_id == 202:
    masks.append(mask)
    # print(mask)
    # print(type(mask))

    # plt.imshow(mask)
    # plt.axis("off")
    # plt.show()

    return raw_img, mask

def image_to_sub_images(coco, img_id):
    # display_image(coco, i)
    color, mask = imgID_to_sample(coco, img_id)
    sub_imgs, sub_masks = split_image_with_mask(color, mask, ignore=[0])
    return sub_imgs, sub_masks

def imgID_to_image(coco, img_id):
    # display_image(coco, i)
    color, mask = imgID_to_sample(coco, img_id)
    return color, mask

my_catIds = [6]
clutter_imgIds = cat_to_images(clutter_coco, my_catIds)
train_imgIds = cat_to_images(coco, my_catIds)
random_imgIds = cat_to_images(random_coco, my_catIds)
light_imgIds = cat_to_images(light_coco, my_catIds)
# imgID_to_image(clutter_coco, imgIds[0])
print(len(clutter_imgIds))


# display_image(coco, train_imgIds[30])

# build object bank
object_bank = {}
for i in range(1, 61):
    imgIds = cat_to_images(coco, [i])
    cur_len = len(imgIds)
    # hard code the sample idx
    # will change later
    cur_idx = [i for i in range(0, cur_len, cur_len//5)]
    sample_imgIds = [imgIds[i] for i in cur_idx]
    object_bank[i] = sample_imgIds


def catID_to_support_set(coco, cat_id, object_bank):
    res_imgs = []
    res_masks = []
    for i in object_bank[cat_id]:
        sub_imgs, sub_masks = image_to_sub_images(coco, i)
        res_imgs += sub_imgs
        res_masks += sub_masks
    return res_imgs, res_masks

def catID_to_clutter_query(clutter_coco, cat_ids):
    clutter_imgIds = cat_to_images(clutter_coco, cat_ids)
    print("number of clutter images: ", len(clutter_imgIds))
    query_img, query_mask = imgID_to_image(clutter_coco, clutter_imgIds[12])
    query_img = np.array(query_img).astype(int)
    return query_img, query_mask

res_imgs, res_masks = catID_to_support_set(coco, 12, object_bank)
query_img, query_mask = catID_to_clutter_query(clutter_coco, [12])

print("number of res images: ", len(res_imgs))
num_support_imgs = 5
# Define a list of numbers
num_list = list(range(0, len(res_imgs)))
# print("num_list: ", num_list)

# Randomly pick 5 distinct numbers from the list
random_numbers = random.sample(num_list, num_support_imgs)
# random_numbers = [0, 1, 2, 3, 4]
sup_imgs = [res_imgs[i] for i in random_numbers]
sup_masks = [res_masks[i] for i in random_numbers]
print("number of support images: ", len(sup_imgs))

first_mask = sup_masks[0]
print("first mask shape: ", first_mask.shape)
frames = []
for i in range(len(sup_imgs)):
    frames.append(sup_imgs[i])
frames.append(query_img)

print("number of frames: ", len(frames))


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
# network = XMem(config, './saves/Apr18_14.21.17_retrain_stage0_only_s0/Apr18_14.21.17_retrain_stage0_only_s0_30625.pth').eval().to(device)

mask = first_mask #np.array(Image.open(mask_name))
# print(np.unique(mask))
print("the unqiue values in the mask: ", np.unique(mask))
num_objects = 1 #len(np.unique(mask)) - 1
masks = sup_masks
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
        if current_frame_index <= 4:
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
        visualization = overlay_davis(frame, prediction, alpha=0.5, fade=False)

        imgplot = plt.imshow(visualization.astype(np.uint8))
        plt.axis("off")
        plt.show()

        current_frame_index += 1
#


