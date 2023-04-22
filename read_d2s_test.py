import json
import os
from pycocotools.coco import COCO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np

# json_path = "./dataset/d2s/d2s_annotations_v1.1/annotations/D2S_training.json"
img_path = "./dataset/d2s/d2s_images_v1/images"

# test_json_path = "./dataset/d2s/d2s_annotations_v1.1/annotations/D2S_validation_clutter.json"
# test_json_path = "./dataset/d2s/d2s_annotations_v1.1/annotations/D2S_validation_random_background.json"
test_json_path = "./dataset/d2s/d2s_annotations_v1.1/annotations/D2S_training.json"
json_path = "./dataset/d2s/d2s_annotations_v1.1/annotations/D2S_training.json"

# json_labels = json.load(open(json_path, 'r'))
# print(json_labels["info"])

# load coco format data
coco = COCO(annotation_file=json_path)

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
print(coco.getCatIds())



# show the first three images
# for img_id in ids[3000:4000]:
#     # get annotations idx
#     ann_ids = test_coco.getAnnIds(imgIds=img_id)
#
#     # according to anno idx, get the annos
#     targets = test_coco.loadAnns(ann_ids)
#     for i in range(len(targets)):
#         # print("found obj id", targets[i]['category_id'])
#         # print("found ", coco_classes[targets[i]['category_id']])
#         if targets[i]['category_id'] == 42:
#             print("found ", coco_classes[42])
#             print("image id: ", img_id)
#         # get image file name
#         # path = test_coco.loadImgs(img_id)[0]["file_name"]
#         # frame = np.array(Image.open(os.path.join(img_path, path)), dtype=float)
#         # frames.append(frame)
#         # # print(frame)
#         # # read image
#         # img = Image.open(os.path.join(img_path, path)).convert('RGB')
#         # draw = ImageDraw.Draw(img)
#         # # draw box to image
#         # for target in targets:
#         #     x,y,w,h = target['bbox']
#         #     x1, y1, x2, y2 = x, y, int(x+w), int(y+h)
#         #     draw.rectangle((x1,y1,x2,y2))
#         #     draw.text((x1, y1), coco_classes[target["category_id"]])
#         # fig = plt.figure()
#         # fig.add_subplot(1,2,1)
#         # plt.imshow(img)
#         # plt.axis("off")
#         # test_coco.showAnns(targets)
#         #
#         # fig.add_subplot(1,2,2)
#         # mask = test_coco.annToMask(targets[0])
#         # for i in range(1, len(targets)):
#         #     mask += test_coco.annToMask(targets[i])
#         # print("image id: ", img_id)
#         #
#         # first_mask = mask[0]
#         # # if img_id == 200 or img_id == 201 or img_id == 202:
#         # masks.append(mask)
#         # # print(mask)
#         # # print(type(mask))
#         # # plt.subplots(122)
#         # plt.imshow(mask)
#         # plt.axis("off")
#         # plt.show()
#
#         # break