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
    plt.imshow(color)
    plt.show()
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
    vis_color_and_mask(sub_images[0], sub_masks[0])
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
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis("off")
    coco.showAnns(targets)

    fig.add_subplot(1, 2, 2)
    mask = coco.annToMask(targets[0])
    for i in range(1, len(targets)):
        mask += coco.annToMask(targets[i]) * (i + 1)
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

    return raw_img, mask

def image_to_sub_images(coco, img_id):
    # display_image(coco, i)
    color, mask = imgID_to_sample(coco, img_id)
    sub_imgs, sub_masks = split_image_with_mask(color, mask, ignore=[0])
    return sub_imgs, sub_masks


my_catIds = [6]
clutter_imgIds = cat_to_images(clutter_coco, my_catIds)
train_imgIds = cat_to_images(coco, my_catIds)
random_imgIds = cat_to_images(random_coco, my_catIds)
light_imgIds = cat_to_images(light_coco, my_catIds)
# imgID_to_image(clutter_coco, imgIds[0])
print(len(clutter_imgIds))

# for i in range(0, 90, 30):
#     display_image(clutter_coco, clutter_imgIds[i])

# display_image(coco, train_imgIds[30])
# display_image(random_coco, random_imgIds[36])
# display_image(light_coco, light_imgIds[0])

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
    # if cur_len == 60:
    #     print("cur_idx: {}".format(cur_idx))
    #     print("sample_imgIds: {}".format(sample_imgIds))

res_imgs = []
res_masks = []


# for i in object_bank[6]:
#     sub_imgs, sub_masks = image_to_sub_images(coco, i)
#     res_imgs += sub_imgs
#     res_masks += sub_masks

image_to_sub_images(coco, object_bank[6][-1])

# print("number of sub images: ", len(res_imgs))
# print("number of sub masks: ", len(res_masks))



