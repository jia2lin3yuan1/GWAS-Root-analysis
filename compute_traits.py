"""Run DeepLab-ResNet on a given image.

This script computes a segmentation mask for a given image.
"""

from __future__ import print_function

import argparse
import os
import pdb
import cv2
import numpy as np

from scipy import ndimage
from scipy import misc as smisc
from skimage import measure as smeasure
from skimage.feature import peak_local_max

from matplotlib import pyplot as plt


def find_y_line(rgbI, plantI,
                lft_size=100, delta_y=2, win_v=10,
                fname=None, saver=None):
    '''
    @func: find the y-line that separate the background and the canvas
           1. convert rgbI to LAB channel and working on A channel
           2. A channel: small to large jump
    '''
    out_y = np.zeros((rgbI.shape[1]), dtype=np.int32) + rgbI.shape[0]

    # step 1: convert to LAB color-space and auto-compute thr
    labI = cv2.cvtColor(rgbI[..., [2,1,0]], cv2.COLOR_BGR2LAB)

    idx_0 = (rgbI.shape[0]*2)//8
    idx_1 = (rgbI.shape[0]*3)//8
    idx_2 = (rgbI.shape[0]*5)//8
    idx_3 = (rgbI.shape[0]*6)//8

    lft_st, lft_end = lft_size//10, lft_size//10+lft_size
    lft_meanA = labI[:, lft_st:lft_end, 1].mean(axis=1)
    sort_meanA = sorted(lft_meanA)
    thrA = (np.mean(sort_meanA[idx_0:idx_1]) + np.mean(sort_meanA[idx_2:idx_3]))/2

    # step 2: binary map morphology transform
    binaryI = (labI[..., 1] > thrA).astype(np.uint8)
    bg_val = 1

    # step 3: on the left side, detect the y line
    for i in range(rgbI.shape[1]):
        if i == 0:
            v_st, v_end = rgbI.shape[0]-1, -1
        else:
            out_y[i] = out_y[i-1]
            v_st, v_end = min(rgbI.shape[0]-1, out_y[i-1]+delta_y), max(0, out_y[i-1]-delta_y-1)
        for k in range(v_st, v_end, -1):
            up_window = binaryI[max(0, k-win_v):min(rgbI.shape[0], k+1), i]
            plant_window = plantI[max(0, k-win_v):min(rgbI.shape[0], k+1), i]
            if plant_window.sum() > 0:
                pass
            elif up_window[-1]==bg_val and up_window.sum()>up_window.shape[0]*0.6:
                out_y[i] = k+1
                break
            else:
                pass
        if i > lft_size and abs(out_y[i] - out_y[0]) > 20:
            out_y[i] = out_y[i-1]

        # for ruler
        if i == 0:
            binaryI[out_y[0]:][labI[out_y[0]:, :, 0] > 150] = 0

    # step 4: construct image
    ret = np.zeros_like(rgbI[...,0], dtype=np.uint8)
    for i in range(rgbI.shape[1]):
        ret[:out_y[i], i] = 1

    # debug vis
    if fname is not None:
        vis_images = [rgbI, labI[...,1]>thrA, ret[..., None]*rgbI, \
                      binaryI[..., None]*rgbI, labI[...,1], labI[...,2]]
        vis_palettes = ['RGB', 'label', 'RGB', 'RGB', 'range', 'range']
        saver.save_group_pilImage_RGB(vis_images, vis_palettes, nr=2, nc=3, resize=512, save_path=fname)

    return ret


def remove_noise(intensityI, flagI, size_thr=None, intensity_thr=None, a2p_thr=None):
    if intensity_thr is None:
        props = smeasure.regionprops(smeasure.label(flagI>0))
    else:
        props = smeasure.regionprops(smeasure.label(flagI>0), intensity_image=intensityI)

    for prop in props:
        coord = prop.coords
        if size_thr is not None and prop.area < size_thr:
            flagI[coord[:,0], coord[:,1]]=0
        if a2p_thr is not None and prop.area< a2p_thr*prop.perimeter:
            flagI[coord[:,0], coord[:,1]]=0
        if intensity_thr is not None and prop.mean_intensity < intensity_thr:
            flagI[coord[:,0], coord[:,1]]=0


def _root_type_basal(rootI, canvas_y, thr=0.2):
    '''
    @func: check type of root, basal -- with blue canvas background,
                               lateral -- with black background
    @param:
    '''
    ringI = cv2.dilate(rootI, np.ones((7, 7))) - cv2.dilate(rootI, np.ones((3,3)))
    canvas_cnt = canvas_y[ringI>0].sum()
    isBasal  = (canvas_cnt < ringI.sum()*thr)
    return isBasal


def _BFS_one_step(step, ys, xs, flagI, trackI, dummyI=None, dirI=None):
    '''
    @func: BFS search to track path from source to all valid pixels in flagI
    @param: step -- depth of current BFS layer
            ys, xs -- node in layer 'step'
            flagI -- 2D array, valid pixels to move
            trackI -- tracking path in BFS
            dummyI -- binary 2D array, valid pixels are path but don't count step.
            dirI -- 2D array, if given, actually perform backward.
                              children pixel should have smaller value than its parent.
    '''
    # one step search
    new_ys, new_xs = [], []
    for y, x in zip(ys, xs):
        # traverse
        for j in range(-1, 2):
            ny = y+j
            if ny < 0 or ny >= flagI.shape[0]:
                continue
            for i in range(-1, 2):
                nx = x+i
                if nx < 0 or nx >= flagI.shape[1]:
                    continue

                # check if the neighbour is valid for moving forward
                if flagI[ny, nx]==0 or (dirI is not None and dirI[ny, nx] > dirI[y, x]):
                    continue
                else:
                    new_ys.append(ny)
                    new_xs.append(nx)
                    flagI[ny, nx], trackI[ny, nx, 0] = 0, step
                    if dummyI is not None:
                        trackI[ny, nx, 1] = trackI[y, x, 1] + dummyI[ny, nx]

    return np.asarray(new_ys), np.asarray(new_xs)


def _BFS_searchPath(ys, xs, flagI, dummyI=None, dirI=None):
    '''
    @func: BFS search to track path from source to all valid pixels in flagI
    @param: ys, xs -- node in layer 'step'
            flagI -- binary 2D array, valid pixels to move
            dummyI -- binary 2D array, valid pixels are path but don't count step.
            dirI -- 2D array, if given, actually perform backward.
                              children pixel should have smaller value than its parent.
    '''
    trackI = np.zeros([flagI.shape[0], flagI.shape[1], 2], dtype = np.float32)
    step = 1

    # update the step 1
    k = np.arange(len(ys))
    flagI[ys[k], xs[k]]     = 0
    trackI[ys[k], xs[k], :] = step

    # start tracking
    while True:
        new_ys, new_xs = _BFS_one_step(step, ys, xs, flagI, trackI, dummyI, dirI)
        ys, xs, step   = new_ys, new_xs, step+1
        if(len(new_ys) == 0):
            break

    if dummyI is None:
        return trackI[..., 0]
    else:
        return trackI[..., 1]

def _backtracking_oneRoot(src_loc, distI, invalidI, spatial_alpha=0.5):
    if (distI * invalidI).max() > 0:
        upd_distI = _BFS_searchPath(src_loc[0], src_loc[1], distI>0, dummyI=(invalidI==0))
    else:
        upd_distI = distI

    # possible path
    poss_pathI = upd_distI * (invalidI==0)
    # padding 1 because peak_local_max couldn't find local maximum along image boundary
    pad_poss_pathI = np.pad(poss_pathI, ((1,1), (1,1)), 'edge')
    coords  = peak_local_max(pad_poss_pathI, min_distance=1, threshold_abs=0)
    if (len(coords)==0):
        return None

    # find endpoint of root, and back-tracking
    locYs, locXs = coords[:, 0]-1, coords[:, 1]-1
    srcY,  srcX  = src_loc[0].mean(), src_loc[1].mean()
    spatial_dist = np.sqrt((locYs-srcY)**2 + (locXs-srcX)**2)
    i = range(len(locYs))
    dist_to_src  = spatial_dist * spatial_alpha + \
                   poss_pathI[locYs[i], locXs[i]] * (1-spatial_alpha)
    sort_idx     = sorted(range(len(dist_to_src)), key=lambda i: -dist_to_src[i])
    ey = locYs[sort_idx[0]]
    ex = locXs[sort_idx[0]]

    # and back-tracking
    pathI  = _BFS_searchPath(np.asarray([ey]), np.asarray([ex]), distI>0, dirI=distI)
    return pathI

def _find_majorRoots(root_distM, src_info, MAX_DIST=1e4):
    '''
    @func: track of major roots.
    @param: root_distM -- 3D array, [num_root, ht, wd]
    '''
    # flagI denote all the tracked root pixels
    flagI     = np.zeros(root_distM.shape[1:])

    # tracking major path, starting from shortest root
    cand_roots = []
    root_size = root_distM.max(axis=-1).max(axis=-1)
    sort_ids = sorted(range(len(root_size)), key=lambda i: -root_size[i])
    for k in sort_ids:
        pathI = _backtracking_oneRoot(src_info[k]['src_loc'], root_distM[k]+0, flagI)
        if pathI is None:
            continue

        pathI_noOccl = pathI * (flagI==0)
        if (pathI_noOccl>0).sum() < (pathI>0).sum()*0.2:
            pass
        else:
            flagI += pathI_noOccl
            rootInfo = {'dist': root_distM[k]*(pathI>0),
                        'eff_dist': pathI_noOccl,
                        'src_size':src_info[k]['src_size'],
                        'dist2stem': src_info[k]['dist2stem']}
            cand_roots.append(rootInfo)

    return {'major_flagI': flagI, 'cands':cand_roots}


def _compute_single_root_attribute(rootInfo, res_rootI, canvas_y):
    '''
    @param: rootInfo -- dict,
            res_rootI -- residual rootI
            rot_rgb -- scalar
    '''
    distI = rootInfo['dist']
    rootI = (rootInfo['dist']>0).astype(np.uint8)
    eff_rootI = (rootInfo['eff_dist']>0).astype(np.uint8)
    src_size  = rootInfo['src_size']
    dist2stem = rootInfo['dist2stem']
    det_root = dict()

    # conditions for a fake root
    if(distI.max() < np.clip(src_size*1.1, 10, max(80, src_size))) or \
        (dist2stem > 30 and rootI.sum() < 200):
        det_root['valid'] = False
    else:
        # a real root
        det_root['valid'] = True
        root_minorI       = (cv2.dilate(eff_rootI, np.ones([3,3])) - eff_rootI)*res_rootI
        src_ys, src_xs    = np.where(root_minorI > 0)
        det_root['dist_minor'] = _BFS_searchPath(src_ys, src_xs, res_rootI)
        det_root['num_minor']  = smeasure.label(root_minorI>0).max()

        full_rootI = rootI + (det_root['dist_minor']>0)
        isBasal    = _root_type_basal(full_rootI, canvas_y)
        det_root['type'] = 'Basal' if isBasal else 'Lateral'

    return det_root


def compute_root_traits(rgbI, rootI, stemI):
    root_ret = {'type': [],
                'length': [],
                'area': [],
                'major':[],
                'minor': [],
                'num_minor':[]}

    # resize if needed
    if rgbI.shape[:2] != rootI.shape[:2]:
        oriHt, oriWd = rgbI.shape[:2]
        scale_h, scale_w = float(oriHt)/rootI.shape[0], float(oriWd)/rootI.shape[1]
        rs_rgbI = cv2.resize(rgbI, (rootI.shape[1], rootI.shape[0]))
    else:
        scale_h, scale_w = 1.0, 1.0
        rs_rgbI = rgbI
    canvas_y   = find_y_line(rs_rgbI, rootI+stemI)
    stem_distI = ndimage.distance_transform_edt(stemI==0)

    # process each single root by disjoint connection to stem and compute root traits
    det_roots = []
    res_rootI = rootI + 0
    while(res_rootI.sum() > 0):
        # BFS tracking all possible major roots
        stem_distI[res_rootI==0] = 0
        dist2stem  = stem_distI[stem_distI>0].min()
        connI      = smeasure.label((stem_distI == dist2stem).astype(np.uint8))
        poss_props = smeasure.regionprops(connI)
        root_distM, src_info = [], []
        for prop in poss_props:
            ys, xs = np.where(connI==prop.label)
            y0,x0,y1,x1 = prop.bbox
            root_distM.append(_BFS_searchPath(ys, xs, res_rootI+0))
            src_info.append({'src_size': max(y1-y0, x1-x0), \
                             'dist2stem': dist2stem, \
                             'src_loc': [ys, xs]})
        major_info = _find_majorRoots(np.asarray(root_distM), src_info)

        # compute root attribute
        res_rootI[major_info['major_flagI']>0]=0
        for rootInfo in major_info['cands']:
            root_attr = _compute_single_root_attribute(rootInfo, res_rootI, canvas_y)
            if root_attr['valid'] == True:
                det_roots.append((rootInfo, root_attr))

    # compute root attribute
    new_rootI = np.zeros(rootI.shape, dtype=np.float32)
    new_root_distI = np.zeros(rootI.shape, dtype=np.float32)
    new_root_typeI = np.zeros(rootI.shape, dtype=np.float32)
    for k, ele in enumerate(det_roots):
        rootInfo, root_attr = ele
        root_ret['type'].append(root_attr['type'])
        root_ret['length'].append(int(rootInfo['dist'].max()*scale_h))
        root_ret['major'].append(int((rootInfo['dist']>0).sum()*scale_h*scale_w))
        root_ret['minor'].append(int((root_attr['dist_minor']>0).sum()*scale_h*scale_w))
        root_ret['area'].append(root_ret['major'][k] + root_ret['minor'][k])
        root_ret['num_minor'].append(root_attr['num_minor'])

        # image
        new_rootI[rootInfo['dist']>0] = k + 1
        new_rootI[root_attr['dist_minor']>0] = k + 1

        new_root_distI += (rootInfo['dist']*255./(rootInfo['dist'].max()+0.01))
        new_root_distI += (root_attr['dist_minor']*255./(root_attr['dist_minor'].max()+0.01))

        new_root_typeI[rootInfo['dist']>0] = 2 if root_attr['type'] == 'Basal' else 1
        new_root_typeI[root_attr['dist_minor']>0] = 4 if root_attr['type'] == 'Basal' else 3

    return {'attr': root_ret,
            'root_img': new_rootI,
            'root_dist_img': new_root_distI,
            'root_type_img': new_root_typeI}


def mean_middle_sector(inVec, st_ratio=1/4., end_ratio=3/4.,):
    if len(inVec)==0:
        return 0

    sort_vec  = sorted(inVec)
    st_idx = int(len(sort_vec)*st_ratio)
    end_idx = int(len(sort_vec)*end_ratio)
    if end_idx <= st_idx:
        st_idx, end_idx = max(0, len(sort_vec)-3), len(sort_vec)

    return np.mean(sort_vec[st_idx:end_idx])

def compute_leaf_area(leafI, ori_shape=None):
    '''
    @param:
    '''
    oriHt, oriWd = ori_shape
    scale_h, scale_w = float(oriHt)/leafI.shape[0], float(oriWd)/leafI.shape[1]
    leaf_area = leafI.sum()

    return {'area': int(leaf_area*scale_h*scale_w)}

def compute_stem_width(stemI, ori_shape=None):
    oriHt, oriWd = ori_shape
    scale_h, scale_w = float(oriHt)/stemI.shape[0], float(oriWd)/stemI.shape[1]

    stem_size = np.sum(stemI, axis=1)
    stem_size = stem_size[stem_size>0]

    return {'width': int(mean_middle_sector(stem_size) * scale_w),
            'area': int(stem_size.sum() * scale_h * scale_w)}


def compute_ruler_size(rulerI, ori_shape, length_thr=100, ruler_cm_wd=3.):
    '''
    @param: rulerI -- detect of ruler on resize shape
            ori_shape -- orginal (ht, wd)
            lenth_thr -- # of column ruler
            ruler_cm_wd -- ruler width in cm
    '''
    oriHt, oriWd = ori_shape
    ruler_vsize = np.sum(rulerI, axis=0)

    if ruler_vsize.shape[0] > length_thr:
        eff_sizes = ruler_vsize[ruler_vsize>0]
        ruler_size = mean_middle_sector(eff_sizes, st_ratio=2/5., end_ratio = 3/5.)
    else:
        ruler_size = 0

    ruler_size *= oriHt/float(rulerI.shape[0])
    ruler_ret   = {'width': int(ruler_size)}
    ruler_ret['pix_cm'] = np.round(ruler_cm_wd/float(ruler_size), 5) if ruler_size>0 else 0
    ruler_ret['num_pix_1cm'] = np.round(float(ruler_size)/ruler_cm_wd, 5)
    return ruler_ret

