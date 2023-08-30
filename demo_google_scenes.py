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
from itertools import product


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
    query_sub_imagess = sub_images
    query_sub_masks = sub_masks
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
        vis_color_and_mask(query_sub_imagess[j], query_sub_masks[j])
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

def get_cur_scene2(scene_dir, subdir):
    cur_scene = {}
    # read scene description
    filename = os.path.join(scene_dir, subdir, 'scene_description.txt')
    print(filename)
    scene_description = json.load(open(filename))
    # print(scene_description.keys())

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
    scene_folder = os.path.join(scene_dir, subdir)
    print("scene_folder: ", scene_folder)
    index = 1 #np.random.randint(0, 7)
    color, depth, label, meta = load_frame_rgbd(scene_folder, index)
    # plt.imshow(color)
    # plt.show()

    sub_images, sub_masks = split_image_with_mask(color, label, ignore=[0, 1])
    # print('mixed mask unique mask 2: ', np.unique(sub_masks[2]))
    vis_color_and_mask(sub_images[1], sub_masks[1])
    query_img = color
    query_masks = sub_masks
    cur_scene['query_img'] = query_img
    cur_scene['query_masks'] = query_masks
    print("query_masks: ", query_masks)
    print("query_masks length: ", len(query_masks))
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
    # print("unique nums: ", np.unique(label))

    support_imgs = []
    support_masks = []

    for j in range(n):
        obj_name = obj_names[j]
        scene_folder = os.path.join(obj_dir, obj_name)

        for template_index in range(9):
            color, depth, label, meta = load_object_rgbd(scene_folder, template_index)
            sub_images, sub_masks = split_image_with_mask(color, label, ignore=[0])
            support_imgs.append(sub_images[0])
            support_masks.append(sub_masks[0])
            # vis_color_and_mask(sub_images[0], sub_masks[0])
            # if obj_name.strip() == 'Elephant':
            #     print("Elephant: ", template_index)
            #     # vis_color_and_mask(sub_images[0], sub_masks[0])
            #     color = sub_images[0]
            #     image = Image.fromarray(color)
            #     # image.save(f'only_objects/Elephant_{template_index}.png')
            #     break


    cur_scene['support_imgs'] = support_imgs
    cur_scene['support_masks'] = support_masks

    return cur_scene


def get_cur_scene_images(scene_dir, subdir, obj_dir):
    cur_scene = {}
    # read scene description
    filename = os.path.join(scene_dir, subdir, 'scene_description.txt')
    scene_description = json.load(open(filename))
    # print(scene_description.keys())

    # get object names
    objects = scene_description['object_descriptions']
    obj_names = []
    for obj in objects:
        mesh_filename = obj['mesh_filename']
        names = mesh_filename.split('/')
        # get object name
        obj_name = names[-3]
        obj_names.append(obj_name)
    print('This scene has objects:', obj_names)
    n = len(obj_names)

    # load one image
    scene_folder = os.path.join(scene_dir, subdir)
    scene_imgs = []
    for scene_idx in range(7):
        color_file = os.path.join(scene_folder, 'rgb_%05d.jpg' % scene_idx)
        # get rid of root folder
        # words = color_file.split('/')
        # color_file = '/'.join(words[2:])
        scene_imgs.append(color_file)
    template_imgs = []
    for j in range(n):
        obj_name = obj_names[j]
        template_folder = os.path.join(obj_dir, obj_name)

        for template_index in range(9):
            template_color_file = os.path.join(template_folder, '%06d-object-rgb.jpg' % template_index)
            # get rid of root folder
            # words = template_color_file.split('/')
            # template_color_file = '/'.join(words[2:])
            template_imgs.append(template_color_file)
    # print("scene_imgs: ", scene_imgs)
    # print('length of scene_imgs: ', len(scene_imgs))
    # print("template_imgs: ", template_imgs)
    # print('length of template_imgs: ', len(template_imgs))

    # get each pair of scene and template
    combinations = list(product(scene_imgs, template_imgs))
    # print("the length of combinations: ", len(combinations))
    # print("combinations: ", combinations[:3])
    return combinations

def scene_img_and_mask(scene_img_path):
    words = scene_img_path.split('/')
    words[-1] = words[-1].replace('rgb', 'segmentation')
    words[-1] = words[-1].replace('jpg', 'png')
    return '/'.join(words)

if __name__ == '__main__':
    root_dir = './FewSOL'
    # scene_dir = root_dir + '/google_scenes/train'
    scene_dir = root_dir + '/google_scenes_demo/train'
    obj_dir = root_dir + '/synthetic_objects'
    
    subdirs = sorted(os.listdir(scene_dir))
    num = len(subdirs)
    print(num)
    print(subdirs)
    #
    # # load mesh names
    # filename = './data/synthetic_objects_folders.txt'
    # meshes = []
    # with open(filename) as f:
    #     for line in f:
    #         meshes.append(line.strip())
    #
    scenes = []
    # for each scene
    for i in range(1):
        get_cur_scene(scene_dir, subdirs[i])
        # scene_temp_pair = get_cur_scene_images(scene_dir, subdirs[i], obj_dir)
        # # plt.show()
        # scenes.append(scene_temp_pair)
        # scene_rgb, template_rgb = scene_temp_pair[0]
        # scene_rbg_path = os.path.join(root_dir, scene_rgb)
        # template_rgb_path = os.path.join(root_dir, template_rgb)
        # print("template_rgb_path: ", template_rgb_path)
        # print("scene_rbg_path: ", scene_rbg_path)
        # scene_mask = scene_img_and_mask(scene_rbg_path)

        # load scene image and mask
        # vis to check
        # scene_img = cv2.imread(scene_rbg_path)
        # scene_img = cv2.cvtColor(scene_img, cv2.COLOR_BGR2RGB)
        # scene_mask = cv2.imread(scene_mask, cv2.IMREAD_ANYDEPTH)
        # print("scene_mask: ", scene_mask)
        # print("scene_mask shape: ", scene_mask.shape)
        # print("scene_mask unique: ", np.unique(scene_mask))
        # plt.imshow(scene_img)
        # plt.show()
        # plt.imshow(scene_mask)
        # plt.show()




