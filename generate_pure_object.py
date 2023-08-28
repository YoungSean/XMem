#!/usr/bin/env python3
import os
import time
import json
import cv2
import scipy.io
import numpy as np
from simulation_util import imread_indexed
from matplotlib import pyplot as plt
from PIL import Image


def vis_color_and_mask(image, mask):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # Display the first image in the first subplot
    axs[0].imshow(image)
    axs[0].set_title('Image')
    # Display the second image in the second subplot
    axs[1].imshow(mask, cmap='gray')
    axs[1].set_title('Its mask')
    plt.show()

def load_object_rgbd(scene_folder, i):
    color_file = os.path.join(scene_folder, '%06d-color.jpg' % i)
    color = cv2.imread(color_file)
    color = np.ascontiguousarray(color[:, :, ::-1])

    depth_file = os.path.join(scene_folder, '%06d-depth.png' % i)
    depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)

    meta_file = os.path.join(scene_folder, '%06d-meta.mat' % i)
    meta = scipy.io.loadmat(meta_file)

    seg_file = os.path.join(scene_folder, '%06d-label-binary.png' % i)
    label = imread_indexed(seg_file)

    return color, depth, label, meta

def split_image_with_mask(color, mask, ignore=None):
    mask_ids = np.sort(np.unique(mask))
    # print("sorted mask ids: ", mask_ids)
    sub_images = []
    sub_masks = []
    for i in mask_ids:
        # i==0, background. i==1, table
        if i in ignore:
            continue
        temp_mask = (mask==i).astype(int)
        sub_image = color.copy()
        sub_mask = mask.copy()
        sub_image[temp_mask==0] = 0
        sub_mask[temp_mask==0] = 0
        sub_images.append(sub_image)
        sub_masks.append(sub_mask)
        # print("sub mask: ", np.unique(sub_mask))

        # visualize the image and its mask
        # vis_color_and_mask(sub_image, sub_mask)
    return sub_images, sub_masks


if __name__ == '__main__':
    root_dir = './FewSOL'
    # scene_dir = root_dir + '/google_scenes/train'
    # scene_dir = root_dir + '/google_scenes_demo/train'
    obj_dir = root_dir + '/synthetic_objects'

    subdirs = sorted(os.listdir(obj_dir))
    print("subdirs: ", subdirs[:2])
    for subdir in subdirs:
        print("subdir: ", subdir)
        for i in range(9):
            # print("subdir: ", subdir, " i: ", i)
            color_file = os.path.join(subdir, '%06d-color.jpg' % i)
            # print("color_file: ", color_file)
            color, depth, label, meta = load_object_rgbd(obj_dir+'/'+subdir, i)
            sub_images, sub_masks = split_image_with_mask(color, label, ignore=[0,1])
            color = sub_images[0]
            image = Image.fromarray(color)
            object_image_path = os.path.join(root_dir + '/synthetic_objects', subdir, '%06d-object-rgb.jpg' % i)
            image.save(object_image_path)
