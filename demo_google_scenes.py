#!/usr/bin/env python3
import os
import time
import json
import cv2
import scipy.io
import numpy as np
from simulation_util import imread_indexed
from matplotlib import pyplot as plt


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

def vis_color_and_mask(image, mask):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # Display the first image in the first subplot
    axs[0].imshow(image)
    axs[0].set_title('Image')
    # Display the second image in the second subplot
    axs[1].imshow(mask, cmap='gray')
    axs[1].set_title('Its mask')
    plt.show()

def split_image_with_mask(color, mask, ignore=None):
    mask_ids = np.sort(np.unique(mask))
    print("sorted mask ids: ", mask_ids)
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


def load_frame_rgbd(scene_folder, i):
    color_file = os.path.join(scene_folder, 'rgb_%05d.jpg' % i)
    color = cv2.imread(color_file)
    color = np.ascontiguousarray(color[:, :, ::-1])

    depth_file = os.path.join(scene_folder, 'depth_%05d.png' % i)
    depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)

    meta_file = os.path.join(scene_folder, 'meta_%05d.mat' % i)
    meta = scipy.io.loadmat(meta_file)

    seg_file = os.path.join(scene_folder, 'segmentation_%05d.png' % i)
    label = imread_indexed(seg_file)

    print('===================================')
    print(color_file)
    print(depth_file)
    print(seg_file)
    print(meta_file)
    print('===================================')

    return color, depth, label, meta

def get_cur_scene(scene_dir, subdir):
    cur_scene = {}
    # read scene description
    filename = os.path.join(scene_dir, subdir, 'scene_description.txt')
    print(filename)

    scene_description = json.load(open(filename))
    print(scene_description.keys())

    # get object names
    objects = scene_description['object_descriptions']
    obj_names = []
    for obj in objects:
        mesh_filename = obj['mesh_filename']
        names = mesh_filename.split('/')
        # get object name
        obj_name = names[-3]
        obj_names.append(obj_name)
    print(obj_names)
    n = len(obj_names)

    # load one image
    scene_folder = os.path.join(scene_dir, subdirs[i])
    index = np.random.randint(0, 7)
    color, depth, label, meta = load_frame_rgbd(scene_folder, index)
    # plt.imshow(color)
    # plt.show()

    sub_images, sub_masks = split_image_with_mask(color, label, ignore=[0, 1])
    # print('mixed mask unique mask 2: ', np.unique(sub_masks[2]))
    # vis_color_and_mask(sub_images[2], sub_masks[2])
    query_img = color
    query_masks = sub_masks
    cur_scene['query_img'] = query_img
    cur_scene['query_masks'] = query_masks
    vis_color_and_mask(color, label)

    # visualization
    # fig = plt.figure()
    # ax = fig.add_subplot(1+n, 3, 1)
    # plt.title('color')
    # plt.imshow(color)
    # ax = fig.add_subplot(1+n, 3, 2)
    # plt.title('depth')
    # plt.imshow(depth)
    # ax = fig.add_subplot(1+n, 3, 3)
    # plt.title('label')
    # plt.imshow(label)
    print("unique nums: ", np.unique(label))

    support_imgs = []
    support_masks = []

    for j in range(n):
        obj_name = obj_names[j]
        scene_folder = os.path.join(obj_dir, obj_name)
        color, depth, label, meta = load_object_rgbd(scene_folder, 0)
        # ax = fig.add_subplot(1+n, 3, 3 + j*3 + 1)
        # plt.imshow(color)
        # plt.imshow(label)
        # print("unique nums of obj1 label: ", np.unique(label))
        # mask = (label > 0).astype(int)
        # print(mask)
        # print("unique nums of obj1 mask: ", np.unique(mask))
        # if j==2:
        #     sub_images, sub_masks = split_image_with_mask(color, label, ignore=[0])
        #     vis_color_and_mask(sub_images[0], sub_masks[0])
        #     break
        sub_images, sub_masks = split_image_with_mask(color, label, ignore=[0])
        support_imgs.append(sub_images[0])
        support_masks.append(sub_masks[0])
        vis_color_and_mask(sub_images[0], sub_masks[0])

        color, depth, label, meta = load_object_rgbd(scene_folder, 1)
        # ax = fig.add_subplot(1+n, 3, 3 + j*3 + 2)
        # plt.imshow(color)
        # plt.imshow(label)
        sub_images, sub_masks = split_image_with_mask(color, label, ignore=[0])
        support_imgs.append(sub_images[0])
        support_masks.append(sub_masks[0])
        vis_color_and_mask(sub_images[0], sub_masks[0])

        color, depth, label, meta = load_object_rgbd(scene_folder, 2)
        # ax = fig.add_subplot(1+n, 3, 3 + j*3 + 3)
        # plt.imshow(color)
        # plt.imshow(label)
        sub_images, sub_masks = split_image_with_mask(color, label, ignore=[0])
        support_imgs.append(sub_images[0])
        support_masks.append(sub_masks[0])
        vis_color_and_mask(sub_images[0], sub_masks[0])

    cur_scene['support_imgs'] = support_imgs
    cur_scene['support_masks'] = support_masks

    return cur_scene

if __name__ == '__main__':

    root_dir = './FewSOL'
    scene_dir = root_dir + '/google_scenes/train'
    obj_dir = root_dir + '/synthetic_objects'
    
    subdirs = sorted(os.listdir(scene_dir))
    num = len(subdirs)
    print(subdirs)

    # load mesh names
    filename = './data/synthetic_objects_folders.txt'
    meshes = []
    with open(filename) as f:
        for line in f:
            meshes.append(line.strip())

    scenes = []
    # for each scene
    for i in range(num):
        cur_scene = get_cur_scene(scene_dir, subdirs[i])
        # plt.show()
        scenes.append(cur_scene)


