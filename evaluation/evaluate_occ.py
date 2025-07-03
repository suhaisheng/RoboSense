import os
import sys
import pickle
import numpy as np
from tqdm import tqdm

def read_occ_gt(input_path):
    with open(input_path, "rb") as f:
        content = pickle.load(f)
    gt_map = {}
    for info in content['infos']:
        if 'occ_gt_path' in info and info['occ_gt_path'] is not None:
            npz_name = os.path.split(info["occ_gt_path"])[-1]
            gt_map[npz_name] = info["occ_gt_path"]
    return gt_map

def IOU_compute_BEV(dt, gt, mask=1):
    dt, gt = np.sum(dt[..., 4:], axis=2)>0, np.sum(gt[..., 4:], axis=2)>0
    dt, gt = dt.astype(np.int8), gt.astype(np.int8)
    dt, gt = dt*mask, gt*mask

    I = (dt * gt).sum()
    U = dt.sum() + gt.sum() - I
    return I / (U + 0.00001)

def IOU_compute_3D(dt, gt, mask=1):
    dt, gt = dt*mask, gt*mask
    dt, gt = dt[..., 4:],gt[..., 4:]
    I = (dt * gt).sum()
    U = dt.sum() + gt.sum() - I
    return I / (U + 0.00001)

def occ_eval(pkl_path, infer_root, distance=[0,12.8]):
    gt_map = read_occ_gt(pkl_path)
    npz_list = os.listdir(infer_root)
    thres = [0.1*i for i in range(10)]
    mIOU_3D = {thre:0.0 for thre in thres}
    mIOU_BEV = {thre:0.0 for thre in thres}
    mask = np.ones((64,64))
    for i in range(64):
        for j in range(64):
            x = (i-32)*0.4
            y = (j-32)*0.4
            if (x**2+y**2)**0.5 >= distance[0] and (x**2+y**2)**0.5 <= distance[1]:
                continue
            else:
                mask[i][j] = 0
    mask_3D = np.expand_dims(mask, axis=2)
    mask_BEV = mask
    print(mask.sum())
    occ_sum = 0
    for npz in tqdm(npz_list):
        dt_path = os.path.join(infer_root, npz)
        try:
            gt_path = gt_map[npz]
        except:
            continue
        dt_float = np.load(dt_path)['semantics']
        # mask = np.argmax(dt_float, axis=-1) == 1
        # dt_float = mask * dt_float[..., 1]
        gt = np.load(gt_path)['semantics']

        for cls in [1]:
            occ_sum += 1
            cls_dt_float = dt_float[..., cls]
            # cls_dt_float = dt_float
            cls_gt = (gt==cls).astype(np.int8)
            for thre in thres:
                cls_dt = (cls_dt_float>thre).astype(np.int8)
                mIOU_3D[thre] += IOU_compute_3D(cls_dt, cls_gt, mask_3D)
                mIOU_BEV[thre] += IOU_compute_BEV(cls_dt, cls_gt, mask_BEV)

    mIOU_3D = {k:v/occ_sum for k,v in mIOU_3D.items()}
    mIOU_BEV = {k:v/occ_sum for k,v in mIOU_BEV.items()}
    print(distance, '\n', mIOU_3D)
    print(distance, '\n', mIOU_BEV)

if __name__ == '__main__':
    pkl_path, infer_root = sys.argv[1:3]
    for distance in [[0,12.8], [0,2], [2,5], [5,12.8]]:
        occ_eval(pkl_path, infer_root, distance)