import json
import os
from pycocotools.coco import COCO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import cv2

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
print(coco.loadCats(coco.getCatIds())[:5])
print(coco.loadCats(5))

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

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
nms=[cat['name'] for cat in cats]
print('Clutter COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('Clutter COCO supercategories: \n{}'.format(' '.join(nms)))

my_catIds = clutter_coco.getCatIds(catNms=['gepa_bio_und_fair_kamillentee'])
my_catIds2 = coco.getCatIds(catNms=['gepa_bio_und_fair_kamillentee'])
print("my_catIds: {}".format(my_catIds))
print("my_catIds2: {}".format(my_catIds2))
def cat_to_images(coco_file, cat_ids):
    imgIds = coco_file.getImgIds(catIds=cat_ids)
    print("imgIds: {}".format(imgIds))
    print("number of images: {}".format(len(imgIds)))
    return imgIds

clutter_imgIds = cat_to_images(clutter_coco, my_catIds)
train_imgIds = cat_to_images(coco, my_catIds)
random_imgIds = cat_to_images(random_coco, my_catIds)
light_imgIds = cat_to_images(light_coco, my_catIds)
# imgID_to_image(clutter_coco, imgIds[0])
display_image(clutter_coco, clutter_imgIds[20])
display_image(coco, train_imgIds[30])
display_image(random_coco, random_imgIds[36])
display_image(light_coco, light_imgIds[0])

#