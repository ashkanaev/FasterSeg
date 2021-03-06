# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys
import numpy as np
from easydict import EasyDict as edict

C = edict()
config = C
cfg = C

C.seed = 12345

"""please config ROOT_dir and user when u first using"""
C.repo_name = 'fasterseg_2'
C.abs_dir = osp.realpath(".")
C.this_dir = C.abs_dir.split(osp.sep)[-1]
C.root_dir = C.abs_dir[:C.abs_dir.index(C.repo_name) + len(C.repo_name)]
C.log_dir = osp.abspath(osp.join(C.root_dir, 'log', C.this_dir))

"""Data Dir"""
C.dataset_path = "/media/fdata/ashkanaev/argo2"
C.img_root_folder = C.dataset_path
C.gt_root_folder = C.dataset_path
C.train_source = osp.join(C.dataset_path, "train.txt")
C.train_eval_source = osp.join(C.dataset_path, "train.txt")
C.eval_source = osp.join(C.dataset_path, "val.txt")
C.test_source = osp.join(C.dataset_path, "val.txt")

"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(osp.join(C.root_dir, 'tools'))

"""Image Config"""
C.num_classes = 8
C.background = 0
C.image_mean = np.array([0.485, 0.456, 0.406])
C.image_std = np.array([0.229, 0.224, 0.225])
# C.target_size = 320
C.down_sampling = (320, 640) # first down_sampling then crop ......
C.gt_down_sampling = 1
C.num_train_imgs = 3000
C.num_eval_imgs = 500

""" Settings for network, this would be different for each kind of model"""
C.bn_eps = 1e-5
C.bn_momentum = 0.1

"""Train Config"""
C.lr = 0.01
C.set_lr = True

C.momentum = 0.9
C.weight_decay = 5e-4
C.nepochs = 600
C.niters_per_epoch = 200
C.num_workers = 6
C.train_scale_array = [0.75, 1, 1.25]

"""Eval Config"""
C.eval_stride_rate = 5 / 6
C.eval_scale_array = [1, ]
C.eval_flip = False
C.eval_base_size = 320
C.eval_crop_size = 320
C.eval_height = 320
C.eval_width = 640



"""Augmentation Config"""
C.aug_prob = 0.95 # set 0 to exclude augmentation

C.hue_saturation_value_prob = 1.0
C.hue_shift_limit = 5
C.sat_shift_limit = 45
C.val_shift_limit = 5
C.random_sized_crop_prob = 0.5
C.min_max_height = 0.8
C.horizontal_flip_prob = 0.5
C.random_gamma_prob = 0.5
C.random_contrast_prob = 0.5
C.random_brightness_prob = 0.5
C.random_contrast_limit = 0.15
C.random_brightness_limit = 0.15
C.one_of_blur_prob = 0.15
C.motion_blur_prob = 0.1
C.median_blur_prob = 0.1
C.median_blur_limit = 1 # type int
C.blur_prob = 0.1
C.blur_limit = 1 # type int
C.clahe_prob = 0.5
C.clahe_limit = 1 # type int
C.iaaemboss_prob = 0.5


C.layers = 16
""" Train Config """
C.mode = "student" # "teacher" or "student"
if C.mode == "teacher":
    ##### train teacher model only ####################################
    C.arch_idx = [0] # 0 for teacher
    C.branch = [2]
    C.width_mult_list = [4./12, 6./12, 8./12, 10./12, 1.,]
    C.stem_head_width = [(1, 1)]
    C.load_w = True
    C.load_path = "search-224x448_F12.L16_batch2-20200427-230502" # path to the searched directory
    C.load_epoch = "last" # "last" or "int" (e.g. "30"): which epoch to load from the searched architecture
    C.batch_size = 12
    C.Fch = 12
    C.image_height = 320
    C.image_width = 640
    C.save = "%dx%d_teacher_batch%d"%(C.image_height, C.image_width, C.batch_size)
elif C.mode == "student":
    ##### train student with KL distillation from teacher ##############
    C.arch_idx = [0, 1] # 0 for teacher, 1 for student
    C.branch = [2, 2]
    C.width_mult_list = [4./12, 6./12, 8./12, 10./12, 1.,]
    C.stem_head_width = [(1, 1), (8./12, 8./12),]
    C.load_w = True
    C.load_path = "search-224x448_F12.L16_batch2-20200427-230502" # path to the searched directory
    C.teacher_path = "train-320x640_teacher_batch12-20200630-035432" # where to load the pretrained teacher's weight
    C.load_epoch = "last" # "last" or "int" (e.g. "30")
    C.batch_size = 12
    C.Fch = 12
    C.image_height = 320
    C.image_width = 640
    C.save = "%dx%d_student_batch%d"%(C.image_height, C.image_width, C.batch_size)

########################################
C.is_test = False # if True, prediction files for the test set will be generated
C.is_eval = True # if True, the train.py will only do evaluation for once
C.eval_path = "train-320x640_student_batch12-20200630-124325" # path to pretrained directory to be e:valuated
