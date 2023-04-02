#!/usr/bin/env python3
import os
from os import path
import time
import json
import cv2
import scipy.io
import numpy as np
from simulation_util import imread_indexed
from matplotlib import pyplot as plt
import random

from dataset.range_transform import im_normalization, im_mean


import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np

from dataset.range_transform import im_normalization, im_mean
from dataset.tps import random_tps_warp
from dataset.reseed import reseed

# random.seed(0)  # make sure the order is the same
# np.random.seed(0)
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
    color = np.ascontiguousarray(color[:, :, ::-1])  # BGR to RGB

    depth_file = os.path.join(scene_folder, 'depth_%05d.png' % i)
    depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)

    meta_file = os.path.join(scene_folder, 'meta_%05d.mat' % i)
    meta = scipy.io.loadmat(meta_file)

    seg_file = os.path.join(scene_folder, 'segmentation_%05d.png' % i)
    label = imread_indexed(seg_file)

    # print('===================================')
    # print(color_file)
    # print(depth_file)
    # print(seg_file)
    # print(meta_file)
    # print('===================================')

    return color, depth, label, meta


class FewSOLDataset(Dataset):
    """
    Get the dataset for few-shot object learning.

    num_frames: number of frames to be sampled from each scene. 4 frames: 3 samples + 1 query. change 3 to 4 to get 4 samples.
    """
    def __init__(self, root_dir, num_frames=4, max_num_obj=1):
        self.num_frames = num_frames
        self.max_num_obj = max_num_obj

        self.root_dir = root_dir + '/FewSOL'
        self.scene_dir = self.root_dir + '/google_scenes/train'
        self.obj_dir = self.root_dir + '/synthetic_objects'

        self.subdirs = sorted(os.listdir(self.scene_dir))
        scene_num = len(self.subdirs) # number of scenes
        scene_image_num = 7 # number of images in each scene
        print(self.subdirs)
        self.num = scene_num * scene_image_num

        # load mesh names
        filename = root_dir + '/data/synthetic_objects_folders.txt'
        meshes = []
        with open(filename) as f:
            for line in f:
                meshes.append(line.strip())

        # Final transform without randomness
        self.final_im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])

        self.final_gt_transform = transforms.Compose([
            transforms.ToTensor(),
        ])


    def get_cur_scene(self, scene_dir, subdir, obj_dir, idx):
        cur_scene = {}
        # read scene description
        filename = os.path.join(scene_dir, subdir, 'scene_description.txt')
        # print(filename)

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
        # print(obj_names)
        n = len(obj_names)

        # load one image
        scene_folder = os.path.join(scene_dir, subdir)
        index = idx % 7
        color, depth, label, meta = load_frame_rgbd(scene_folder, index)
        color_file = os.path.join(scene_folder, 'rgb_%05d.jpg' % index)
        # plt.imshow(color)
        # plt.show()

        sub_images, sub_masks = split_image_with_mask(color, label, ignore=[0, 1])
        # print('mixed mask unique mask 2: ', np.unique(sub_masks[2]))
        # vis_color_and_mask(sub_images[2], sub_masks[2])
        query_img = color
        query_masks = sub_masks
        cur_scene['query_img'] = self.final_im_transform(query_img)
        cur_scene['query_masks'] = query_masks
        # vis_color_and_mask(color, label)

        support_imgs = []
        support_masks = []

        cur_object_idx = np.random.randint(n)
        cur_scene['query_mask'] = self.final_gt_transform(query_masks[cur_object_idx])
        obj_name = obj_names[cur_object_idx]
        scene_folder = os.path.join(obj_dir, obj_name)

        num_obj_imgs = 2 #np.random.randint(1,10)
        # Define a list of numbers
        num_list = list(range(0, 9))
        # print("num_list: ", num_list)

        # Randomly pick 5 distinct numbers from the list
        random_numbers = random.sample(num_list, num_obj_imgs)
        # print("random_numbers: ", random_numbers)
        for i in random_numbers:
            color, depth, label, meta = load_object_rgbd(scene_folder, i)
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
            this_img = self.final_im_transform(sub_images[0])
            this_gt = self.final_gt_transform(sub_masks[0])
            support_imgs.append(this_img)
            support_masks.append(this_gt)
            # print("the shape of this_img: ", this_img.shape)
            # print("the shape of this_gt: ", this_gt.shape)
            # vis_color_and_mask(sub_images[0], sub_masks[0])
            # vis_color_and_mask(this_img, this_gt)
            #
            # color, depth, label, meta = load_object_rgbd(scene_folder, 1)
            # sub_images, sub_masks = split_image_with_mask(color, label, ignore=[0])
            # support_imgs.append(sub_images[0])
            # support_masks.append(sub_masks[0])
            # # vis_color_and_mask(sub_images[0], sub_masks[0])
            #
            # color, depth, label, meta = load_object_rgbd(scene_folder, 2)
            # sub_images, sub_masks = split_image_with_mask(color, label, ignore=[0])
            # support_imgs.append(sub_images[0])
            # support_masks.append(sub_masks[0])
            # # vis_color_and_mask(sub_images[0], sub_masks[0])

        cur_scene['support_imgs'] = support_imgs
        cur_scene['support_masks'] = support_masks



        return cur_scene, color_file

    def __getitem__(self, idx):
        additional_objects = np.random.randint(self.max_num_obj)
        indices = [idx, *np.random.randint(self.__len__(), size=additional_objects)]

        # scene folder
        subdir = self.subdirs[indices[0] // 7]
        cur_scene, color_file = self.get_cur_scene(self.scene_dir, subdir, self.obj_dir, idx)

        # cur_scene = self.scenes[indices[0]]
        support_imgs = cur_scene['support_imgs']
        # print("support_imgs type: ", type(support_imgs[0]))
        support_masks = cur_scene['support_masks']
        query_img = cur_scene['query_img']
        query_mask = cur_scene['query_mask']

        num_objects = len(query_mask)  # number of objects in the query image, each object has a mask
        # we just need one object
        # cur_object_idx = np.random.randint(num_objects)
        cur_sample = {}
        cur_sample['query_img'] = query_img
        cur_sample['object_img'] = support_imgs #[3*cur_object_idx:3*cur_object_idx+3]
        cur_sample['object_mask'] = support_masks #[3*cur_object_idx:3*cur_object_idx+3]
        cur_sample['query_mask'] = query_mask #[cur_object_idx]

        info = {}
        info['name'] = color_file  # to be changed
        info['num_objects'] = 1  # to be changed

        # 1 if object exist, 0 otherwise
        selector = [1 if i < info['num_objects'] else 0 for i in range(self.max_num_obj)]
        selector = torch.FloatTensor(selector)

        # print("cur_sample: ", cur_sample)
        # Convert the NumPy arrays to PyTorch tensors
        # tensor_list = [torch.permute(torch.from_numpy(arr), (2, 0, 1)) for arr in cur_sample['object_img']]
        # tensor_list.append(torch.permute(torch.from_numpy(cur_sample['query_img']), (2, 0, 1)))
        tensor_list = [arr for arr in cur_sample['object_img']]
        tensor_list.append(cur_sample['query_img'])
        merged_images = torch.stack(tensor_list, dim=0)
        # first_frame_gt = cur_sample['object_mask'][0]

        # mask_list = [torch.unsqueeze(torch.from_numpy(arr), 0) for arr in cur_sample['object_mask']]
        mask_list = [arr for arr in cur_sample['object_mask']]
        mask_list.append(cur_sample['query_mask'])
        cls_gt = torch.stack(mask_list, dim=0)
        cls_gt[cls_gt > 0] = 1
        # print("the values of cls_gt: ", torch.unique(cls_gt))
        # print("the shape of cls_gt: ", cls_gt.shape)
        # vis_color_and_mask(torch.permute(merged_images[0], (1, 2, 0)), cls_gt[0,0])
        # vis_color_and_mask(torch.permute(merged_images[1], (1, 2, 0)), cls_gt[1, 0])
        # vis_color_and_mask(torch.permute(merged_images[2], (1, 2, 0)), cls_gt[2, 0])
        # vis_color_and_mask(torch.permute(merged_images[-1], (1, 2, 0)), cls_gt[-1, 0])
        # print("the shape of merged_images: ", merged_images.shape)
        # print("the shape of cls_gt: ", cls_gt.shape)
        # print("the shape of cls_gt[:-1]: ", cls_gt[:-1,:,:,:].shape)

        # last_frame_gt = cls_gt[-1, 0]
        # print("the shape of last_frame_gt: ", last_frame_gt.shape)
        # vis_color_and_mask(torch.permute(merged_images[-1], (1, 2, 0)), last_frame_gt)
        # print("the shape of merged_images: ", merged_images.shape)

        data = {
            'rgb': merged_images.float(),  # (T, C, H, W) (3, 3, 384, 384)
            'first_frame_gt': cls_gt[:-1,:,:,:].long(), # get rid of the last frame, (T-1, 1, H, W) (2, 1, 384, 384)
            #first_frame_gt,  # (1, 1, H, W) (1, 1, 384, 384)
            'cls_gt': cls_gt.long(),  # (T, 1, H, W) (3, 1, 384, 384)
            'selector': selector,
            'info': info
        }

        return data



    def __len__(self):
        return self.num




if __name__ == '__main__':
    f_dataset = FewSOLDataset('..')
    # print(f_dataset.scenes[0]['support_imgs'][0].shape)
    print(f_dataset[1]['info'])

    # root_dir = '../FewSOL'
    # scene_dir = root_dir + '/google_scenes/train'
    # obj_dir = root_dir + '/synthetic_objects'
    #
    # subdirs = sorted(os.listdir(scene_dir))
    # num = len(subdirs)
    # print(subdirs)
    #
    # # load mesh names
    # filename = '../data/synthetic_objects_folders.txt'
    # meshes = []
    # with open(filename) as f:
    #     for line in f:
    #         meshes.append(line.strip())
    #
    # scenes = []
    # for each scene
    # for i in range(num):
    #     cur_scene = get_cur_scene(scene_dir, subdirs[i], obj_dir)
    #     # plt.show()
    #     scenes.append(cur_scene)


