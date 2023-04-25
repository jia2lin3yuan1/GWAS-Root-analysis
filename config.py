import os
import numpy as np
from glob import glob
from easydict import EasyDict as edict

cfg = edict()
cfg.root_path     = '/media/yuanjial/LargeDrive/DataSet/Forest-Root-Exp/wk5_Rooting'
cfg.data_ext      = '.JPG'

# step 1, rotation and flip
cfg.preprocess = edict()
cfg.preprocess.data_dir = os.path.join(cfg.root_path, 'images')
cfg.preprocess.data_ext = cfg.data_ext
cfg.preprocess.resize_shape = [256, 256]
cfg.preprocess.save_dir = os.path.join(cfg.root_path, 'step1_preprocess')
cfg.preprocess.save_ext = '.jpg'

# step 2, 2-level semantic segmentation
#                 1).  bg/plant/ruler/label segmentation
#                 2).  bg/stem/leaf/root segmentation
cfg.segment = edict()
cfg.segment.rgb_dir = cfg.preprocess.save_dir
cfg.segment.rgb_ext = cfg.preprocess.save_ext
cfg.segment.sem_1st_dir = os.path.join(cfg.root_path, 'step2_1st_segment')
cfg.segment.sem_1st_ext = '_prob.png'
cfg.segment.sem_2nd_dir = os.path.join(cfg.root_path, 'step2_2nd_segment')
cfg.segment.sem_2nd_ext = '_prob.png'

cfg.segment.sem_1st_classes = {'bg': 0, 'plant':1, 'label':2, 'ruler': 3}
cfg.segment.sem_2nd_classes = {'bg': 0, 'leaf':1, 'stem':2, 'root':3}

cfg.segment.save_dir = os.path.join(cfg.root_path, 'step2_final_segment')
cfg.segment.save_grid_ext  = '_grid.png'
cfg.segment.save_final_ext = '_final.png'

# step3, compute traits based on segmentation result
cfg.analysis = edict()
cfg.analysis.rgb_dir = cfg.preprocess.save_dir
cfg.analysis.rgb_ext = cfg.preprocess.save_ext
cfg.analysis.sem_1st_dir = cfg.segment.sem_1st_dir
cfg.analysis.sem_1st_ext = cfg.segment.sem_1st_ext
cfg.analysis.sem_2nd_dir = cfg.segment.sem_2nd_dir
cfg.analysis.sem_2nd_ext = cfg.segment.sem_2nd_ext
cfg.analysis.step2_dir = cfg.segment.save_dir
cfg.analysis.step2_ext = cfg.segment.save_final_ext

cfg.analysis.sem_1st_classes = cfg.segment.sem_1st_classes
cfg.analysis.sem_2nd_classes = cfg.segment.sem_2nd_classes
cfg.analysis.save_dir = os.path.join(cfg.root_path, 'step3_analysis_traits')


def get_image_list(data_dir, data_ext):
    dir_list = os.listdir(data_dir)

    image_set_index = []
    for fdir in dir_list:
        glob_imgs = glob(os.path.join(data_dir, fdir, '*'+data_ext))

        # get the image name in sorted order
        tmp_imgs  = sorted([ele.split('_') for ele in glob_imgs])
        glob_imgs = ['_'.join(ele) for ele in tmp_imgs]

        img_list = [(fdir, os.path.basename(v)) for v in glob_imgs]
        image_set_index += img_list

    return image_set_index

def parse_filename(fname, ext='.jpg'):
    if isinstance(fname, tuple):
        sub_dir, img_name = fname
    else:
        sub_dir, img_name = '', fname
    img_name = img_name.split(ext)[0]
    return sub_dir, img_name

def makedir(path):
    # exist_ok only work for python3.2 +
    os.makedirs(path, exist_ok=True)

