from __future__ import print_function

import os
import cv2
import numpy as np
import scipy.misc as smisc

from step2_segment_ms import zoomIn_to_oneClass, read_parse_network_segment
from compute_traits import compute_root_traits, compute_stem_width, compute_leaf_area
from compute_traits import compute_ruler_size
from save_tool import SaveTool

from config import get_image_list, parse_filename, makedir
from config import cfg

from matplotlib import pyplot as plt
import pdb

def save_failed_file(fcsv, idx, sub_dir, img_name, message):
    cur_str = '\n' + str(idx) +',' + sub_dir + ',' + img_name + ',' + str(0) + ',' + message
    fcsv.write(cur_str)


def save_effective_file(fcsv, idx, sub_dir, img_name,
                        ruler_ret, stem_ret, leaf_ret, root_ret,
                        img_text=None):
    # text to save in csv file
    fname = str(idx) +',' + sub_dir + ',' + img_name
    ruler_info = str(ruler_ret['pix_cm']) +',' + \
                 str(ruler_ret['num_pix_1cm']) + ','+\
                 str(ruler_ret['width'])
    stem_info  = str(stem_ret['width']) + ',' +\
                 str(stem_ret['area'])
    leaf_info  = str(leaf_ret['area'])
    csv_str    = '\n' + fname + ',' + ruler_info + ','+stem_info + ',' + leaf_info

    root_type       = (' : ').join(root_ret['type'])
    root_len        = (' : ').join([str(ele) for ele in root_ret['length']])
    root_area       = (' : ').join([str(ele) for ele in root_ret['area']])
    root_major      = (' : ').join([str(ele) for ele in root_ret['major']])
    root_minor      = (' : ').join([str(ele) for ele in root_ret['minor']])
    root_num_minor  = (' : ').join([str(ele) for ele in root_ret['num_minor']])
    csv_str         = ','.join([csv_str, root_type, root_len, root_num_minor,\
                                root_area, root_major, root_minor])

    fcsv.write(csv_str)

    # text to show on image
    if img_text is not None:
        img_text = []
        text = 'ruler:: pix (cm) = ' + str(ruler_ret['pix_cm']) +\
                     ', #pix/cm  = ' + str(ruler_ret['num_pix_1cm']) + \
                        ', width = ' + str(ruler_ret['width'])
        img_text.append(text)
        text = 'stem:: len = ' +  str(stem_ret['width']) + \
                  ',  area = ' + str(stem_ret['area'])
        img_text.append(text)
        text = 'leaf:: area = '+ str(leaf_ret['area'])
        img_text.append(text)

        txt_wd = 10
        insert = '-'
        text ='rt-cluster'.center(txt_wd, insert) + ' * ' + \
                    'type'.center(txt_wd, insert) + ' | ' + \
                     'len'.center(txt_wd, insert) + ' | ' + \
               'num-minor'.center(txt_wd, insert) + ' | ' + \
                    'area'.center(txt_wd, insert) + ' | ' + \
                   'major'.center(txt_wd, insert) + ' | ' + \
                   'minor'.center(txt_wd, insert) + ' | '
        img_text.append(text)
        for k in range(len(root_ret['type'])):
            text =                   str(k+1).center(txt_wd, insert) + ' * ' + \
                          root_ret['type'][k].center(txt_wd, insert) + ' | ' + \
                   str(root_ret['length'][k]).center(txt_wd, insert) + ' | ' + \
                str(root_ret['num_minor'][k]).center(txt_wd, insert) + ' | ' + \
                     str(root_ret['area'][k]).center(txt_wd, insert) + ' | ' + \
                    str(root_ret['major'][k]).center(txt_wd, insert) + ' | ' + \
                    str(root_ret['minor'][k]).center(txt_wd, insert) + ' | '
            img_text.append(text)

    return img_text


def main(option, imgSaver):
    fcsv = None
    header = 'No#, sub_directory, file name, pixel size, num pixel in 1cm, ruler width,  \
              stem width, stem area, leaf area, \
              root_type, root lengthes, num minors, root areas, root major areas, root minor areas'

    # image to be process.
    imageList = get_image_list(option.rgb_dir, option.rgb_ext)
    pre_subdir = 'None'
    for k, fname in enumerate(imageList):
        sub_dir, img_name = parse_filename(fname, option.rgb_ext)
        #if 'Phase5' not in sub_dir:
        if (sub_dir!='6.5_Rooting'or 'BESC-1107_2294' not in img_name) and False:
            continue
        print("img {:d} | {:d}, {}".format(k, len(imageList), sub_dir+'/'+img_name))

        if fcsv is None or sub_dir != pre_subdir:
            result_path = os.path.join(option.save_dir, sub_dir)
            makedir(result_path)
            if fcsv is not None:
                fcsv.close()
            pre_subdir = sub_dir
            csvFname = os.path.join(option.save_dir, sub_dir, 'computed_traits.csv')
            fcsv  = open(csvFname, 'w')
            fcsv.write(header)

        # read in data
        rgb_img = cv2.imread(os.path.join(option.rgb_dir, sub_dir, img_name+option.rgb_ext))
        rgb_img = rgb_img[..., [2,1,0]]
        sem_1st_prob =  read_parse_network_segment(img_name,
                                                    os.path.join(option.sem_1st_dir, sub_dir),
                                                    option.sem_1st_ext,
                                                    num_classes=len(option.sem_1st_classes.keys()))
        sem_2nd_prob =  read_parse_network_segment(img_name,
                                                    os.path.join(option.sem_2nd_dir, sub_dir),
                                                    option.sem_2nd_ext,
                                                    num_classes=len(option.sem_2nd_classes.keys()))
        semI_2nd = smisc.imread(os.path.join(option.step2_dir,
                                        sub_dir, img_name+option.step2_ext), mode='P')
        if len(semI_2nd.shape) != 2:
            print('------', fname, semI_2nd.shape)

        # crop zoomin box
        sem_1st_plantI, zoomIn_ret = zoomIn_to_oneClass(rgb_img, sem_1st_prob,
                                                         channel=option.sem_1st_classes['plant'])

        # phenotype segment
        rulerI = (sem_1st_plantI==option.sem_1st_classes['ruler']).astype(np.uint8)
        stemI = (semI_2nd==option.sem_2nd_classes['stem']).astype(np.uint8)
        leafI = (semI_2nd==option.sem_2nd_classes['leaf']).astype(np.uint8)
        rootI = (semI_2nd==option.sem_2nd_classes['root']).astype(np.uint8)

        # compute traits
        if rulerI.sum()==0:
            message = 'no detection of ruler'
            save_failed_file(fcsv, k, sub_dir, img_name, message)
        elif(stemI.sum()==0):
            message = 'no detection of stem'
            save_failed_file(fcsv, k, sub_dir, img_name, message)
        else:
            ruler_ret = compute_ruler_size(rulerI, rgb_img.shape[:2], ruler_cm_wd=2.3)
            stem_ret  = compute_stem_width(stemI, ori_shape=zoomIn_ret['rgb'].shape[:2])
            leaf_ret  = compute_leaf_area(leafI, ori_shape=zoomIn_ret['rgb'].shape[:2])
            root_ret  = compute_root_traits(zoomIn_ret['rgb'], rootI, stemI)

            # save csv file
            img_vis_text = save_effective_file(fcsv, k, sub_dir, img_name, \
                                ruler_ret, stem_ret, leaf_ret, root_ret['attr'], img_text=True)

            # save new result
            scale0 = 10
            saveI = semI_2nd.astype(np.float32)*scale0 + 0
            new_rootI = root_ret['root_img']
            saveI[semI_2nd==option.sem_2nd_classes['root']] = 0
            if new_rootI.max() > 0:
                scale = (254 - option.sem_2nd_classes['root']*scale0)/(new_rootI.max() + 1.0)
                saveI[new_rootI>0] =  new_rootI[new_rootI>0]*scale + option.sem_2nd_classes['root']
            else:
                saveI[0,0] = 254

            save_distI = semI_2nd.astype(np.float32) * 100
            save_distI[semI_2nd==option.sem_2nd_classes['root']] = 0
            save_distI[new_rootI>0] = root_ret['root_dist_img'][new_rootI>0]

            save_typeI = semI_2nd.astype(np.float32) + 0
            save_typeI[semI_2nd==option.sem_2nd_classes['root']] = 0
            save_typeI[new_rootI>0] =  root_ret['root_type_img'][new_rootI>0] + \
                                        option.sem_2nd_classes['root']

            vis_images = [rgb_img, sem_1st_plantI,
                          zoomIn_ret['rgb'], saveI,
                          save_distI, save_typeI,
                          np.zeros_like(sem_1st_plantI)]
            vis_palettes = ['RGB', 'label', 'RGB', 'range', 'range', 'label','label']
            texts = ['RGB', 'plant/ruler/label', 'zoomIn_rgb',
                     'leaf/stem/root', 'root path', 'root-type', img_vis_text]
            imgSaver.save_group_pilImage_RGB(vis_images, vis_palettes, texts, nr=4, nc=2,
                    resize=256, save_path=os.path.join(result_path, img_name+'_final.png'))

            saveI = semI_2nd.astype(np.float32) + 0
            new_rootI = root_ret['root_img']
            saveI[semI_2nd==option.sem_2nd_classes['root']] = 0
            saveI[new_rootI>0] =  new_rootI[new_rootI>0] + option.sem_2nd_classes['root']
            imgSaver.save_single_pilImage_gray(saveI, palette='label', save_path=os.path.join(result_path, img_name+'_zm_sem.png'))

    if fcsv is not None:
        fcsv.close()

if __name__ == '__main__':
    imgSaver = SaveTool()

    main(cfg.analysis, imgSaver)
