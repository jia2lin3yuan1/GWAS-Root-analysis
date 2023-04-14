import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

from config import get_image_list, parse_filename, makedir
from config import cfg
import pdb

def read_and_preprocess(fpath, resize_shape=None):
    img = cv2.imread(fpath)
    oriHt, oriWd = img.shape[:2]

    # resize and rotate image to ruler/label on the right-bottom area
    if resize_shape != None:
        rs_ht, rs_wd = resize_shape
        rs_img = cv2.resize(img, (rs_wd, rs_ht))
    else:
        rs_img = img

    _, pro_mode = preprocess_detect(rs_img, canvas_thr=30)

    img = encode_preprocess(img, pro_mode)
    return {'rgb': img, 'mode':pro_mode}

def preprocess_detect(img, canvas_thr=10, cn_size=80):
    '''
    @param: cn_size -- corner size
            img -- [ht, wd, ch], in 'RGB'
    '''
    def _median3(a, b, c):
        maxV = max(max(a,b), c)
        minV = min(min(a,b), c)
        return a+b+c - maxV - minV

    top_lft_corner = img[:cn_size, :cn_size, :]
    top_rht_corner = img[:cn_size, -cn_size:, :]
    bot_lft_corner = img[-cn_size:, :cn_size, :]
    bot_rht_corner = img[-cn_size:, -cn_size:, :]

    # based on assumption blue channel mean: label > fg_canvas > bg
    mean_blue_val = [ np.mean(top_lft_corner[..., 0]),  # blue channel
                      np.mean(top_rht_corner[..., 0]),
                      np.mean(bot_lft_corner[..., 0]),
                      np.mean(bot_rht_corner[..., 0]) ]
    sort_idx = sorted(range(4), key=lambda i: mean_blue_val[i])
    bg_idxs_sum = sort_idx[0] + sort_idx[1]
    if bg_idxs_sum == 1:
        # top_lft, top_rht are both background
        if sort_idx[-1] == 3:
            # label on the bot_rht corner
            proMode = {}
        else:
            # label on the bot_lft corner
            proMode = {'hflip':True}

    elif bg_idxs_sum == 5:
        # top_lft, top_rht are both foreground
        # label must on the top_rht corner
        assert(sort_idx[-1] ==1)
        proMode = {'vflip':True}

    elif bg_idxs_sum == 4:
        # top_rht, bot_rht are both background
        # label on the bot_lft corner
        assert(sort_idx[-1] ==2)
        proMode = {'hflip':True, 'transpose':True}

    elif  bg_idxs_sum == 2:
        # top_lft, bot_lft are both background
        # label on the top_rht corner
        assert(sort_idx[-1] ==1)
        proMode = {'vflip':True, 'transpose':True}
    else:
        proMode = {}

    return img, proMode


def preprocess_detect_bk(img, canvas_thr=10, cn_size=30):
    '''
    @param: cn_size -- corner size
            img -- [ht, wd, ch], in 'RGB'
    '''
    def _median3(a, b, c):
        maxV = max(max(a,b), c)
        minV = min(min(a,b), c)
        return a+b+c - maxV - minV

    top_lft_corner = img[:cn_size, :cn_size, :]
    top_rht_corner = img[:cn_size, -cn_size:, :]
    bot_lft_corner = img[-cn_size:, :cn_size, :]
    bot_rht_corner = img[-cn_size:, -cn_size:, :]

    top_lft = np.mean(top_lft_corner)
    top_rht = np.mean(top_rht_corner)
    bot_lft = np.mean(bot_lft_corner)
    bot_rht = np.mean(bot_rht_corner)

    bg_uplft_I = np.mean(np.max(top_lft_corner, axis=-1)) - _median3(top_rht, bot_lft, bot_rht)
    bg_uprht_I = np.mean(np.max(top_rht_corner, axis=-1)) - _median3(top_lft, bot_lft, bot_rht)

    if(bg_uplft_I > bg_uprht_I+canvas_thr):
        img     = np.transpose(np.fliplr(img), [1,0,2])
        proMode = {'hflip':True, 'transpose':True}
    elif(bg_uplft_I+canvas_thr < bg_uprht_I):
        img     = np.transpose(np.flipud(img), [1,0,2])
        proMode = {'vflip':True, 'transpose':True}
    elif(bg_uplft_I>canvas_thr and  bg_uprht_I > canvas_thr):
        img     = np.flipud(img)
        proMode = {'vflip':True}
    else:
        proMode = {}

    return img, proMode

def encode_preprocess(img, proMode):
    if 'hflip' in proMode:
        img = np.fliplr(img)
    if 'vflip' in proMode:
        img = np.flipud(img)
    if 'transpose' in proMode:
        if img.ndim==2:
            img = np.transpose(img, [1,0])
        else:
            img = np.transpose(img, [1,0, 2])

    return img

def decode_preprocess(img, proMode):
    if 'transpose' in proMode:
        if img.ndim ==2:
            img   = np.transpose(img, [1,0])
        else:
            img   = np.transpose(img, [1,0,2])
    if 'hflip' in proMode:
        img   = np.fliplr(img)
    if 'vflip' in proMode:
        img   = np.flipud(img)

    return img


def main(option):
    image_set_index = get_image_list(option.data_dir, option.data_ext)
    for k, fname in enumerate(image_set_index):
        sub_dir, img_name = parse_filename(fname, option.data_ext)
        print("img {:d} | {:d}, {}".format(k, len(image_set_index), sub_dir+'/'+img_name))

        # read image and preprocess
        fpath = os.path.join(option.data_dir, sub_dir, img_name+option.data_ext)
        ret = read_and_preprocess(fpath, option.resize_shape)

        # save new iamge
        save_path = os.path.join(option.save_dir, sub_dir)
        makedir(save_path)

        cv2.imwrite(os.path.join(save_path, img_name+option.save_ext), ret['rgb'])

if __name__ == "__main__":
    main(cfg.preprocess)


