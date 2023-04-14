"""Run DeepLab-ResNet on a given image.

This script computes a segmentation mask for a given image.
    -- scipy.misc could read label image for scipy <= 1.20: scipy.misc.imread(fname, mode='P')
    -- new tool for label image reading is: skimage.io.imread(fname, pilmode='P')
"""

from __future__ import print_function
import copy
import os
import heapq
import cv2
import numpy as np
import scipy.misc as smisc
from scipy import ndimage
from skimage import measure as smeasure
from PIL import Image

from config import get_image_list, parse_filename, makedir
from config import cfg
from save_tool import SaveTool
from pymeanshift import segment as mf_segment
from compute_traits import mean_middle_sector, remove_noise

from matplotlib import pyplot as plt
import pdb


def extract_bbox(labelI, idx=1):
    '''
    @ get the bbox of given label on the image.
    '''
    bbox = None
    props = smeasure.regionprops(labelI.astype(np.uint8))
    for prop in props:
        if prop.label == idx:
            bbox = [prop.bbox[0], prop.bbox[1], prop.bbox[2], prop.bbox[3]]
            break

    # if empty, locate on center
    if bbox is None:
        ht, wd = labelI.shape
        bbox = [ht//4, wd//4, (ht*3)//4, (wd*3)//4]

    return bbox


def zoomIn_extend_bbox(ori_rgb, bbox, bbox_shape=None, min_size_ratio=0.2, ext_scale=1.2):
    '''
    @func: crop local area w.r.t. bbox. Note that the bbox might not in same shape as ori_rgbI
    @param: bbox_shape -- if not None, the [ht, wd] of the image bbox is from.
    '''
    ht, wd = ori_rgb.shape[:2]
    if bbox_shape is not None:
        scale_h = float(ht)/bbox_shape[0]
        scale_w = float(wd)/bbox_shape[1]
    else:
        scale_h, scale_w = 1.0, 1.0
    y0,x0,y1,x1 = bbox
    ny0, nx0 = int(np.floor(y0*scale_h)), int(np.floor(x0*scale_w))
    ny1, nx1 = int(np.ceil(y1*scale_h)), int(np.ceil(x1*scale_w))

    # extend window and keep image ht-wd-ratio
    nht = min(ht, (ny1-ny0) * ext_scale)
    nwd = min(wd, (nx1-nx0) * ext_scale)
    if nht < (ht*nwd)//wd:
        nht = (ht*nwd)//wd # need to improve nht
    else:
        nwd = (wd*nht)//ht # need to improve nht
    if nht < ht *min_size_ratio:
        nht, nwd = ht*min_size_ratio, wd*min_size_ratio
    nht, nwd = int(nht), int(nwd)

    # locate new bbox
    cy, cx = int((ny0+ny1)//2), int((nx0+nx1)//2)
    ny0, nx0 = int(cy - nht//2), int(cx - nwd//2)
    ny1, nx1 = int(cy + nht//2), int(cx + nwd//2)
    if ny0 < 0:
        by0, by1 = 0, nht
    elif ny1 >= ht:
        by0, by1 = ht-nht, ht
    else:
        by0,by1 = ny0, ny1

    if nx0 < 0:
        bx0, bx1 = 0, nwd
    elif nx1 >= wd:
        bx0, bx1 = wd-nwd, wd
    else:
        bx0, bx1 = nx0, nx1

    # crop to get the zoomIn area
    zoomin_ret = {'size':[int(nht), int(nwd)],
                  'rgb': ori_rgb[by0:by1, bx0:bx1],
                  'rgb_box': [by0, bx0, by1, bx1],
                  'label_box':[int(by0/scale_h), int(bx0/scale_w), \
                               int(by1/scale_h), int(bx1/scale_w)]}
    return zoomin_ret

def zoomIn_to_oneClass(rgb_img, sem_prob, channel=1, prob_thr=0.05, bbox_min_size_ratio=0.2):
    sem_labelI = semantic_label(sem_prob,
                                special_ch=[channel],
                                special_thr=[prob_thr],
                                keep_conns=1)
    ori_bbox   = extract_bbox(sem_labelI, idx=channel)
    zoomIn_ret = zoomIn_extend_bbox(rgb_img, ori_bbox,
                                    sem_labelI.shape,
                                    min_size_ratio=bbox_min_size_ratio)

    return sem_labelI, zoomIn_ret


def isolated_segment_based_color(rgb_img, sem_1stI, plant_idx=1):
    '''
    @func: segment out salient foreground based on color
    @param: rgb_img -- image in [RGB] mode
            sem_1stI  -- arr in [ht, wd], plant/label/ruler foreground image. same size as rgb_img
            plant_idx -- int, label val of plant in sem_1stI
    '''
    rgb_img  = rgb_img.astype(np.float32)
    ht, wd   = rgb_img.shape[:2]

    # rgb image preprocess
    bgI         = (sem_1stI == 0).astype(np.uint8)
    rg_img      = rgb_img[..., :2]
    rg_bg_img   = rg_img * bgI[..., None]
    rg_bg_val   = rg_bg_img.sum(axis=1)/(bgI[..., None].sum(axis=1)+1)
    salientI_h  = (rg_img - rg_bg_val[:, None, :]).max(axis=-1) - rgb_img[..., 2]
    salientI_h[salientI_h < 0]   = 0
    salientI_h[salientI_h > 255] = 255

    # meanshift segment
    mf_spatial_radius = 11
    mf_range_radius   = 9
    mf_min_density    = 10
    mf_rgb_img  = rgb_img * (salientI_h[..., None]>0)
    _, mf_labelI, _   = mf_segment((mf_rgb_img).astype(np.uint8),
                                   mf_spatial_radius, mf_range_radius, mf_min_density)
    mf_labelI = mf_labelI + 1

    # combine salientI and meanshift result to remove noise
    fg_salient_thr = 10
    sure_bgI = np.logical_and(salientI_h <= fg_salient_thr*1.5, sem_1stI!=plant_idx)
    other_semI  = np.logical_and(sem_1stI>0, sem_1stI!=plant_idx)
    props = smeasure.regionprops(mf_labelI, intensity_image=salientI_h)
    cnt   = 1
    for prop in props:
        coord = prop.coords
        invalid_cnt   = (sure_bgI[coord[:,0], coord[:,1]]>0).sum()
        other_sem_cnt = (other_semI[coord[:,0], coord[:,1]]>0).sum()
        if invalid_cnt > prop.area*0.5 or other_sem_cnt > prop.area*0.1 or \
           (prop.area < 20 and prop.mean_intensity < fg_salient_thr):
            mf_labelI[coord[:,0], coord[:,1]]      = 0
        else:
            mf_labelI[coord[:,0], coord[:,1]] = cnt
            cnt += 1
    salientI_h[mf_labelI == 0] = 0

    return {'segment': mf_labelI,
            'salient': salientI_h.astype(np.uint8)}


def find_semId(semSegI, prop):
    coord = prop.coords
    unq_ids, unq_cnts = np.unique(semSegI[coord[:,0], coord[:,1]], return_counts=True)
    max_semId, max_cnt = 0, 0
    for v, c in zip(unq_ids, unq_cnts):
        if v != 0 and c > max_cnt:
            max_semId, max_cnt = v, c

    return max_semId

def loop_detect_via_distance(seedI, candI, flagI, props):
    '''
    @func: seedI growing based on distance step by step on candidate I.
    @params:seedI -- seed image
            candI -- candidate I, pixel's value is the label of prop.
            flagI -- flag for pixels satisfy some condition
            props -- region proposals from candI
    '''
    # construct prop_dict
    prop_dict = dict()
    for prop in props:
        coord = prop.coords
        prop_dict[prop.label]           = prop
        prop_dict[prop.label].eff_ratio = flagI[coord[:,0], coord[:,1]].sum()/float(prop.area)

    # find seeds to be extended
    moveI = copy.deepcopy(candI)
    flagI = (cv2.dilate(seedI, np.ones([3,3], dtype=np.uint8)))*(moveI>0)
    sy, sx = np.where(flagI>0)
    moveI[flagI>0] = 0
    while len(sy)>0:
        for y, x in zip(sy, sx):
            # check if the prop of pix (y, x) is to be growed
            if seedI[y, x]>0:
                continue
            prop = prop_dict[candI[y,x]]
            if prop.eff_ratio < 0.3:
                continue

            coord = prop.coords
            seedI[coord[:,0], coord[:,1]] = 1

        # new growing seeds
        flagI = (cv2.dilate(seedI, np.ones([3,3], dtype=np.uint8)))*(moveI>0)
        sy, sx = np.where(flagI>0)
        moveI[flagI>0] = 0

    return seedI


def _compute_stem_width_height(stemI):
    stem_wd = np.sum(stemI, axis=1)
    if stem_wd.max()>0:
        stem_wd = mean_middle_sector(stem_wd[stem_wd>0])
    else:
        stem_wd = 30

    stem_ht = np.sum(stemI, axis=0)
    if stem_ht.max()>0:
        stem_ht = mean_middle_sector(stem_ht[stem_ht>0])
    else:
        stem_ht = 100
    return int(stem_wd), int(stem_ht)


def refine_leaf_stem_root(sem_prob, colorseg_info, option,
                          root_thrL=0.05, root_salient_thr=5, root_area_thr=10):
    '''
    @func: refining the sem2nd_ret['sem'] w.r.t. color_segmentI
    @params:
           sem_prob   -- array in [ht, wd, ch], probability map
           colorseg_info -- dict including 'segment'|'salient' |'props'
           color_props -- segments based on color information.
    '''
    ht, wd = sem_prob.shape[:2]
    new_labelI = np.zeros([ht, wd])

    # prepare data
    color_salientI = colorseg_info['salient']
    #color_segI = copy.deepcopy(colorseg_info['segment'])
    sem_labelI = semantic_label(sem_prob,
                                 special_ch=[option.sem_2nd_classes['root']],
                                 special_thr=[root_thrL],
                                 keep_conns=-1)
    sem_leafI  = (sem_labelI==option.sem_2nd_classes['leaf']).astype(np.uint8)
    sem_rootI  = (sem_labelI==option.sem_2nd_classes['root']).astype(np.uint8)
    sem_stemI  = (sem_labelI==option.sem_2nd_classes['stem']).astype(np.uint8)

    # voting
    for k in colorseg_info['props']:
        prop = colorseg_info['props'][k]
        coord = prop.coords
        leaf_cnt = float(sem_leafI[coord[:,0], coord[:,1]].sum())
        stem_cnt = float(sem_stemI[coord[:,0], coord[:,1]].sum())
        root_cnt = float(sem_rootI[coord[:,0], coord[:,1]].sum())
        if (leaf_cnt>prop.area*0.3 and stem_cnt < leaf_cnt):
            new_labelI[coord[:,0], coord[:,1]] = option.sem_2nd_classes['leaf']
        elif root_cnt > prop.area*0.1 and stem_cnt < root_cnt:
            new_labelI[coord[:,0], coord[:,1]] = option.sem_2nd_classes['root']
        elif stem_cnt > prop.area*0.3:
            new_labelI[coord[:,0], coord[:,1]] = option.sem_2nd_classes['stem']
        else:
            pass

    # remove false root pixels and root
    sem_rootI_h = (sem_prob.argmax(axis=-1) == option.sem_2nd_classes['root'])
    det_rootI = new_labelI == option.sem_2nd_classes['root']
    new_labelI[det_rootI>0] = 0
    det_rootI[color_salientI<root_salient_thr] = 0
    props = smeasure.regionprops(smeasure.label(det_rootI))
    for prop in props:
        coord = prop.coords
        if sem_rootI_h[coord[:,0], coord[:,1]].sum() > prop.area*0.3 and \
           prop.area > root_area_thr:
            new_labelI[coord[:,0], coord[:,1]] = option.sem_2nd_classes['root']

    return new_labelI


def trick_steel_falseRoot(labelI, option):
    rootI = labelI == option.sem_2nd_classes['root']
    rootI = cv2.morphologyEx(rootI.astype(np.uint8),
                             cv2.MORPH_CLOSE,
                             np.ones([13, 13], dtype=np.uint8))
    props = smeasure.regionprops(smeasure.label(rootI))
    if(len(props) ==0):
        return labelI

    # stem y
    stemY, _ = np.where(labelI == option.sem_2nd_classes['stem'])
    y_thr = stemY.min()*0.2 + stemY.max()*0.8

    # remove steel
    prop_info  = [(prop, prop.bbox[0]) for prop in props]
    sorted_idx = sorted(range(len(prop_info)), key=lambda i: prop_info[i][1])
    idx = sorted_idx[0]
    if prop_info[idx][1] < y_thr:
        coord = prop_info[idx][0].coords
        labelI[coord[:,0], coord[:,1]] = 0

    return labelI

def trick_leaf_falseRoot(labelI, option):
    # empty image
    eff_classes = np.unique(labelI)
    if eff_classes.max()==0 or \
       option.sem_2nd_classes['root'] not in eff_classes or \
       option.sem_2nd_classes['stem'] not in eff_classes:
        return

    # has or has no leaf
    if option.sem_2nd_classes['leaf'] in eff_classes:
        leafI      = labelI == option.sem_2nd_classes['leaf']
        leaf_distI = ndimage.distance_transform_edt(leafI==0)
    else:
        leaf_distI = None

    # stem and root
    stemI      = labelI == option.sem_2nd_classes['stem']
    stem_distI = ndimage.distance_transform_edt(stemI==0)
    stemY, _   = np.where(stemI==1)
    y_thr      = stemY.min() * 0.7 + stemY.max()*0.3

    rootI = labelI == option.sem_2nd_classes['root']
    rootI = cv2.morphologyEx(rootI.astype(np.uint8),
                             cv2.MORPH_CLOSE,
                             np.ones([11, 11], dtype=np.uint8))

    # check each connected root
    props = smeasure.regionprops(smeasure.label(rootI))

    # remove root not connect to stem but are close to leaf or on high area of the image
    for prop in props:
        coord = prop.coords
        y1 = prop.bbox[2]
        stem_dist = stem_distI[coord[:,0], coord[:,1]].min()
        if leaf_distI is not None:
            leaf_dist = leaf_distI[coord[:,0], coord[:,1]].min()
        else:
            leaf_dist = 1e5

        if stem_dist > min(leaf_dist*3+30, 80) or y1 < y_thr:
            labelI[coord[:,0], coord[:,1]] = 0


def read_parse_network_segment(img_name, file_dir, file_ext, num_classes=4):
    I = np.asarray(Image.open(os.path.join(file_dir, img_name+file_ext)))
    ht, wd = I.shape
    probM = I.reshape([ht, num_classes, wd//num_classes]).transpose([0, 2, 1])
    probM = probM.astype(np.float32)/255.

    return probM


def semantic_label(probM, special_ch=[1], special_thr=[0.1], keep_conns=-1):
    labelI = probM.argmax(axis=-1)

    # for the special channel, remain label if prob > thr
    for ch, thr in zip(special_ch, special_thr):
        extI = (probM[..., ch] > thr) * (labelI==0)
        labelI[extI>0] = ch

    # in 1st semantic, only one isolated region for each class
    if keep_conns > 0:
        for cls in range(1, probM.shape[-1]):
            props = smeasure.regionprops(smeasure.label(labelI==cls))
            if len(props) <= keep_conns:
                continue
            prop_info  = [(prop, prop.area) for prop in props]
            sorted_idx = sorted(range(len(prop_info)), key=lambda i: -prop_info[i][1])
            for k in sorted_idx[keep_conns:]:
                coord = prop_info[k][0].coords
                labelI[coord[:,0], coord[:,1]] = 0
    return labelI

def main(option, imgSaver, RUN_COLOR_SEG=False, RUN_REFINE=False):
    # image to be process.
    imageList = get_image_list(option.rgb_dir, option.rgb_ext)
    for k, fname in enumerate(imageList):
        sub_dir, img_name = parse_filename(fname, option.rgb_ext)
        # if sub_dir!='.5_Rooting':
        if 'Phase5' not in sub_dir:
            pass #continue
        print("img {:d} | {:d}, {}".format(k, len(imageList), sub_dir+'/'+img_name))
        result_path = os.path.join(option.save_dir, sub_dir)
        makedir(result_path)

        # read rgb image and sem_1st
        rgb_img = cv2.imread(os.path.join(option.rgb_dir, sub_dir, img_name+option.rgb_ext))
        rgb_img = rgb_img[..., [2,1,0]]
        sem_1st =  read_parse_network_segment(img_name,
                                              os.path.join(option.sem_1st_dir, sub_dir),
                                              option.sem_1st_ext,
                                              num_classes=len(option.sem_1st_classes.keys()))
        # zoomin cropping
        sem_1st_plantI, zoomIn_ret = zoomIn_to_oneClass(rgb_img, sem_1st,
                                                         channel=option.sem_1st_classes['plant'])

        # sem_2nd
        if RUN_COLOR_SEG or RUN_REFINE:
            sem_2nd =  read_parse_network_segment(img_name,
                                                  os.path.join(option.sem_2nd_dir, sub_dir),
                                                  option.sem_2nd_ext,
                                                  num_classes=len(option.sem_2nd_classes.keys()))
            rgb_img = cv2.resize(zoomIn_ret['rgb'],
                                 (sem_2nd.shape[1], sem_2nd.shape[0]))

            by0, bx0, by1, bx1 = zoomIn_ret['label_box']
            zoomIn_plantI = cv2.resize(sem_1st_plantI[by0:by1, bx0:bx1],
                                       (sem_2nd.shape[1], sem_2nd.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)

        else:
            cv2.imwrite(os.path.join(result_path, img_name+'_zm.jpg'),
                        zoomIn_ret['rgb'][..., [2,1,0]])

        if RUN_COLOR_SEG:
            # color based segment
            colorseg_ret = isolated_segment_based_color(rgb_img, zoomIn_plantI,
                                                        option.sem_1st_classes['plant'])
            imgSaver.save_single_pilImage_gray(colorseg_ret['segment'], 'label',
                        save_path=os.path.join(result_path, img_name+'_clrSeg.png'))

            imgSaver.save_single_pilImage_gray(colorseg_ret['salient'], 'range', autoScale=False,
                        save_path=os.path.join(result_path, img_name+'_clrSalient.png'))
        elif RUN_REFINE: # bypass sure-part to save time
            colorseg_ret = dict()
            clrseg_path = os.path.join(result_path, img_name+'_clrSeg.png')
            colorseg_ret['segment'] = smisc.imread(clrseg_path, mode='P')
            clrsalient_path = os.path.join(result_path, img_name+'_clrSalient.png')
            colorseg_ret['salient'] = smisc.imread(clrsalient_path, mode='P').astype(np.float32)

        if RUN_REFINE:
            # get props
            colorseg_ret['segment'] = smeasure.label(colorseg_ret['segment'])
            props = smeasure.regionprops(colorseg_ret['segment'])
            colorseg_ret['props'] = dict()
            for prop in props:
                colorseg_ret['props'][prop.label] = prop

            # refinement 2nd-level segmentation
            sem_2nd_label = sem_2nd.argmax(axis=-1)
            final_semI = refine_leaf_stem_root(sem_2nd,
                                               colorseg_ret,
                                               option)
            trick_leaf_falseRoot(final_semI, option)

            # save result
            vis_images = [rgb_img, zoomIn_plantI, colorseg_ret['salient'],
                          colorseg_ret['segment'], sem_2nd_label, final_semI]
            vis_palettes = ['RGB', 'label', 'range', 'label', 'label', 'label']
            imgSaver.save_group_pilImage_RGB(vis_images,vis_palettes,nr=3,nc=2,resize=[256, 384],
                            save_path=os.path.join(result_path, img_name+option.save_grid_ext))

            imgSaver.save_single_pilImage_gray(final_semI, 'label',
                       save_path=os.path.join(result_path, img_name+option.save_final_ext))


if __name__ == '__main__':
    imgSaver = SaveTool()

    main(cfg.segment,
         imgSaver,
         RUN_COLOR_SEG=True,
         RUN_REFINE=True)


