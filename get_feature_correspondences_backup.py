# Note: this file should be used inside LoFTR folder
# Usage: python get_feature_correspondences.py
# This file is used to get feature correspondences between a template image and a scene image
# The template image is an object image, and the scene image is a scene image
# git clone the_repo_of_LoFTR to get the LoFTR folder under $ROOT_DIR

import torch
import cv2
import numpy as np
import matplotlib.cm as cm
from matplotlib import pyplot as plt

from src.utils.plotting import make_matching_figure
from src.loftr import LoFTR, default_cfg
import os
from demo_google_scenes import get_cur_scene_images, scene_img_and_mask
from PIL import Image
import scipy.io



file_path = '../FewSOL/google_scenes/palette.txt'  # Replace with your file path
colors_list = []

with open(file_path, 'r') as file:
    for line in file:
        # Remove newline character and append the line to the list
        color_values = line.strip().split()
        color_values = [int(color) for color in color_values]
        colors_list.append(color_values)


# Change the image type here.
# image_type = 'indoor'
image_type = 'outdoor'

# The default config uses dual-softmax.
# The outdoor and indoor models share the same config.
# You can change the default values like thr and coarse_match_type.
matcher = LoFTR(config=default_cfg)
if image_type == 'indoor':
  matcher.load_state_dict(torch.load("weights/indoor_ds_new.ckpt")['state_dict'])
elif image_type == 'outdoor':
  matcher.load_state_dict(torch.load("weights/outdoor_ds.ckpt")['state_dict'])
else:
  raise ValueError("Wrong image_type is given.")
matcher = matcher.eval().cuda()

root_dir = '../FewSOL'
# scene_dir = root_dir + '/google_scenes/train'
scene_dir = root_dir + '/google_scenes_demo/train'
obj_dir = root_dir + '/synthetic_objects'

subdirs = sorted(os.listdir(scene_dir))
num = len(subdirs)
print(num)
# print(subdirs)
#
# # load mesh names
# filename = './data/synthetic_objects_folders.txt'
# meshes = []
# with open(filename) as f:
#     for line in f:
#         meshes.append(line.strip())
#
def get_obj_color_value(template_rgb_path, scene_rgb_path):
    mat_file = scene_rgb_path.replace('rgb', 'meta').replace('jpg', 'mat')
    meta = scipy.io.loadmat(mat_file)
    # print("meta: ", meta)
    foler_name = template_rgb_path.split('/')[-2]
    idx = meta[foler_name][0][0]
    return colors_list[idx]

scenes = []
# for each scene
for i in range(1):
    scene_temp_pair = get_cur_scene_images(scene_dir, subdirs[i], obj_dir)
    # plt.show()
    scenes.extend(scene_temp_pair)
    scene_rgb, template_rgb = scene_temp_pair[1]
    template_rgb_path = os.path.join(root_dir, template_rgb)
    print("template_rgb_path: ", template_rgb_path)
    scene_rgb_path = os.path.join(root_dir, scene_rgb)
    print("scene_rbg_path: ", scene_rgb_path)
    scene_mask = scene_img_and_mask(scene_rgb_path)
    print('my color', get_obj_color_value(template_rgb_path, scene_rgb_path))

print("scenes: ", scenes)
print("the length of scenes: ", len(scenes))
# img0_pth = "../FewSOL/synthetic_objects/Elephant/000000-object-rgb.jpg"
# img1_pth = "../FewSOL/google_scenes_demo/train/scene_00000/rgb_00001.jpg"
img0_pth = template_rgb_path
img1_pth = scene_rgb_path

def check_point_in_object_img(point, mask_img_path):
    # point: [x, y]
    x, y = point
    mask_image = cv2.imread(mask_img_path)

    # print("mask_image shape: ", mask_image.shape)
    print(x,y)
    x = int(x)
    y = int(y)
    pixel_value = mask_image[y, x]
    print("Pixel value at point ({}, {}): {}".format(x, y, pixel_value))
    # Compare the pixel value with (10, 10, 10)
    # avoid noisy pixels from the object image
    good_point = (pixel_value > 10).all()
    if not good_point:
        return False
    return True

def check_point_with_mask(point, mask_img_path, mask_color_value):
    # point: [x, y]
    x, y = point
    mask_image = cv2.imread(mask_img_path)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
    # plt.imshow(mask_image)
    # plt.show()

    # print("mask_image shape: ", mask_image.shape)
    print(x,y)
    x = round(x)
    y = round(y)
    pixel_value = mask_image[y, x]
    print("Pixel value at point ({}, {}): {}".format(x, y, pixel_value.tolist()))
    print("mask_color_value: ", mask_color_value)
    if pixel_value.tolist() == mask_color_value:
        print("good point")
        return True
    else:
        print("not good point")
        return False

def stack_images(image_paths):
    imgs0 = []
    imgs1 = []
    for img0_path, img1_path in image_paths:
        img = cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (640, 480))
        img = torch.from_numpy(img)[None].cuda() / 255.
        imgs0.append(img)
        img = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        print("before resize img: ", img.shape)
        img = cv2.resize(img, (640, 480))
        img = torch.from_numpy(img)[None].cuda() / 255.
        print("after resize img: ", img.shape)
        imgs1.append(img)
    img0_batch_tensor = torch.stack(imgs0)
    img1_batch_tensor = torch.stack(imgs1)
    return img0_batch_tensor, img1_batch_tensor

def imgs2batch(img0_pth, img1_pth):
    image_pair = [img0_pth, img1_pth]
    img0_raw = cv2.imread(image_pair[0], cv2.IMREAD_GRAYSCALE)
    img1_raw = cv2.imread(image_pair[1], cv2.IMREAD_GRAYSCALE)
    img0_raw = cv2.resize(img0_raw, (640, 480))
    img1_raw = cv2.resize(img1_raw, (640, 480))

    img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
    img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
    print("img0: ", img0.shape)
    batch = {'image0': img0, 'image1': img1}
    return batch, img0_raw, img1_raw

batch, img0_raw, img1_raw = imgs2batch(img0_pth, img1_pth)
# Inference with LoFTR and get prediction
with torch.no_grad():
    matcher(batch)
    mkpts0 = batch['mkpts0_f'].cpu().numpy()
    mkpts1 = batch['mkpts1_f'].cpu().numpy()
    mconf = batch['mconf'].cpu().numpy()

# Draw
color = cm.jet(mconf, alpha=0.7)
text = [
    'LoFTR',
    'Matches: {}'.format(len(mkpts0)),
]
fig = make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, mkpts0, mkpts1, text)

# A high-res PDF will also be downloaded automatically.
make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, mkpts0, mkpts1, text, path="LoFTR-colab-demo.pdf")
# print("mkpts0: ", mkpts0)
# print("mkpts1: ", mkpts1)
print("the length of mkpts0: ", len(mkpts0))
print("the length of mkpts1: ", len(mkpts1))



for point in mkpts1:
    # print(point)
    # check_point_in_object_img(point, img0_pth)
    # check_point_in_object_img([0,0], template_rgb_path)
    # print(scene_mask)
    mask_image = cv2.imread(scene_mask)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
    plt.imshow(mask_image)
    unique = np.unique(mask_image)
    # print(np.unique(mask_image))
    plt.show()
    color_value = get_obj_color_value(img0_pth, img1_pth)
    # check_point_with_mask([0,0], scene_mask)
    check_point_with_mask(point, scene_mask, color_value)

