import numpy as np
import os
import pickle as pkl
from collections import defaultdict
from tqdm import tqdm
import json
import torch
from scipy.optimize import linear_sum_assignment


def assign_pred_to_gt_vip3d(
        bbox_result,
        gt_bbox,
        gt_label,
        match_dis_thresh=2.0
    ):
    """Assign pred boxs to gt boxs according to object center preds in lcf.
    Args:
        bbox_result (dict): Predictions.
            'boxes_3d': (LiDARInstance3DBoxes)
            'scores_3d': (Tensor), [num_pred_bbox]
            'labels_3d': (Tensor), [num_pred_bbox]
            'trajs_3d': (Tensor), [fut_ts*2]
        gt_bboxs (LiDARInstance3DBoxes): GT Bboxs.
        gt_label (Tensor): GT labels for gt_bbox, [num_gt_bbox].
        match_dis_thresh (float): dis thresh for determine a positive sample for a gt bbox.

    Returns:
        matched_bbox_result (np.array): assigned pred index for each gt box [num_gt_bbox].
    """     
    matched_bbox_result = torch.ones(
        (len(gt_bbox['center'])), dtype=torch.long) * -1  # -1: not assigned
    gt_centers = torch.from_numpy(gt_bbox['center'][:, :2])
    pred_centers = torch.from_numpy(bbox_result['boxes_3d'][:, :2])
    dist = torch.linalg.norm(pred_centers[:, None, :] - gt_centers[None, :, :], dim=-1)
    # dynamic_list = [0,1,3,4,6,7,8]
    # pred_not_dyn = [label not in dynamic_list for label in bbox_result['labels_3d']]
    # gt_not_dyn = [label not in dynamic_list for label in gt_label]
    # dist[pred_not_dyn] = 1e6
    # dist[:, gt_not_dyn] = 1e6
    dist[dist > match_dis_thresh] = 1e6

    r_list, c_list = linear_sum_assignment(dist)

    for i in range(len(r_list)):
        if dist[r_list[i], c_list[i]] <= match_dis_thresh:
            matched_bbox_result[c_list[i]] = r_list[i]

    return matched_bbox_result

def compute_motion_metric_vip3d(
        gt_bbox,
        gt_label,
        gt_attr_label,
        pred_bbox,
        matched_bbox_result,
        match_dis_thresh=2.0,
    ):
    """Compute EPA metric for one sample.
    Args:
        gt_bboxs (LiDARInstance3DBoxes): GT Bboxs.
        gt_label (Tensor): GT labels for gt_bbox, [num_gt_bbox].
        pred_bbox (dict): Predictions.
            'boxes_3d': (LiDARInstance3DBoxes)
            'scores_3d': (Tensor), [num_pred_bbox]
            'labels_3d': (Tensor), [num_pred_bbox]
            'trajs_3d': (Tensor), [fut_ts*2]
        matched_bbox_result (np.array): assigned pred index for each gt box [num_gt_bbox].
        match_dis_thresh (float): dis thresh for determine a positive sample for a gt bbox.

    Returns:
        EPA_dict (dict): EPA metric dict of each cared class.
    """
    motion_cls_names = ['car', 'pedestrian', 'cyclist', 'total']
    motion_metric_names = ['gt', 'cnt_ade', 'cnt_fde', 'hit',
                            'fp', 'ADE', 'FDE', 'MR']
    
    metric_dict = {}
    for met in motion_metric_names:
        for cls in motion_cls_names:
            metric_dict[met+'_'+cls] = 0.0

    for i in range(pred_bbox['labels_3d'].shape[0]):
        box_name = pred_bbox['labels_3d'][i].lower()
        if i not in matched_bbox_result:
            metric_dict['fp_'+box_name] += 1
            metric_dict['fp_total'] += 1


    for i in range(gt_label.shape[0]):
        box_name = gt_label[i].lower()
        gt_fut_masks = gt_attr_label[i][fut_ts*2:fut_ts*3]
        num_valid_ts = sum(gt_fut_masks==1)
        if num_valid_ts == fut_ts:
            metric_dict['gt_'+box_name] += 1
            metric_dict['gt_total'] += 1

        if matched_bbox_result[i] >= 0 and num_valid_ts > 0:
            metric_dict['cnt_ade_'+box_name] += 1
            metric_dict['cnt_ade_total'] += 1

            m_pred_idx = matched_bbox_result[i]
            gt_fut_trajs = gt_attr_label[i][:fut_ts*2].reshape(-1, 2)
            gt_fut_trajs = gt_fut_trajs[:num_valid_ts]
            pred_fut_trajs = pred_bbox['trajs_3d'][m_pred_idx].reshape(fut_mode, fut_ts, 2)
            pred_fut_trajs = pred_fut_trajs[:, :num_valid_ts, :]
            gt_fut_trajs = gt_fut_trajs.cumsum(dim=-2)
            pred_fut_trajs = pred_fut_trajs.cumsum(dim=-2)
            gt_fut_trajs = gt_fut_trajs + gt_bbox['center'][i, :2]
            pred_fut_trajs = pred_fut_trajs + pred_bbox['boxes_3d'][int(m_pred_idx), :2]

            dist = torch.linalg.norm(gt_fut_trajs[None, :, :] - pred_fut_trajs, dim=-1)
            ade = dist.sum(-1) / num_valid_ts
            ade = ade.min()  # minimum ade across different modality (fut_mode)

            metric_dict['ADE_'+box_name] += ade
            metric_dict['ADE_total'] += ade

            if num_valid_ts == fut_ts:
                fde = dist[:, -1].min()
                metric_dict['cnt_fde_'+box_name] += 1
                metric_dict['cnt_fde_total'] += 1
                metric_dict['FDE_'+box_name] += fde
                metric_dict['FDE_total'] += fde
                if fde <= match_dis_thresh:
                    metric_dict['hit_'+box_name] += 1
                    metric_dict['hit_total'] += 1
                else:
                    metric_dict['MR_'+box_name] += 1
                    metric_dict['MR_total'] += 1

    return metric_dict


def evaluate(results):
    result_metric_names = ['EPA', 'ADE', 'FDE', 'MR']
    motion_cls_names = ['car', 'pedestrian', 'cyclist', 'total']
    motion_metric_names = ['gt', 'cnt_ade', 'cnt_fde', 'hit',
                            'fp', 'ADE', 'FDE', 'MR']
    all_metric_dict = {}
    for met in motion_metric_names:
        for cls in motion_cls_names:
            all_metric_dict[met+'_'+cls] = 0.0
    result_dict = {}
    for met in result_metric_names:
        for cls in motion_cls_names:
            result_dict[met+'_'+cls] = 0.0

    alpha = 0.5
    # fix eval error
    # for i in range(len(results)):
    #     for key in all_metric_dict.keys():
    #         all_metric_dict[key] += results[i]['metric_results'][key]
    for i in range(len(results)):
        for key in all_metric_dict.keys():
            all_metric_dict[key] += results[i][key]
    
    for cls in motion_cls_names:
        result_dict['EPA_'+cls] = (all_metric_dict['hit_'+cls] - \
                alpha * all_metric_dict['fp_'+cls]) / (all_metric_dict['gt_'+cls] + 1e-6)
        result_dict['ADE_'+cls] = all_metric_dict['ADE_'+cls] / (all_metric_dict['cnt_ade_'+cls] + 1e-6)
        result_dict['FDE_'+cls] = all_metric_dict['FDE_'+cls] / (all_metric_dict['cnt_fde_'+cls] + 1e-6)
        result_dict['MR_'+cls] = all_metric_dict['MR_'+cls] / (all_metric_dict['cnt_fde_'+cls] + 1e-6)
    
    print('\n')
    print('-------------- Motion Prediction --------------')
    for k, v in result_dict.items():
        print(f'{k}: {v}')

def motion_traj_generate(track_stat,gt_locations,gt_track_id,velo_path_list,frame_id):
    gt_attr_label = list()
    gt_vel = list()
    for box_id in range(len(gt_locations)):
        cur_track_id = gt_track_id[box_id]
        cur_gt_center = gt_locations[box_id][:2]  # x,y

        left_frame = min(fut_ts, len(velo_path_list) - frame_id - 1)
        # tmp_fut_mask = [1] * left_frame + [0] * (fut_ts - left_frame)
        tmp_fut_mask = [0] * fut_ts
        tmp_fut_traj = np.zeros((fut_ts, 2))
        vel_tag = True
        for left_id in range(left_frame):
            next_gt_anno = track_stat[velo_path_list[frame_id+1+left_id]]
            next_ego2global_rotation = np.array(next_gt_anno['ego2global_rotation'])
            next_ego2global_translation = np.array(next_gt_anno['ego2global_translation']).reshape(3, 1)
            next_ego2global = np.concatenate((next_ego2global_rotation, next_ego2global_translation), -1)

            if cur_track_id in next_gt_anno['id']:
                tmp_fut_mask[left_id] = 1
                next_gt_index = next_gt_anno['id'].tolist().index(cur_track_id)
                next_gt_location = np.array(next_gt_anno['location'][next_gt_index]).reshape(3, 1)
                next_gt_location = np.concatenate((next_gt_location, np.ones((1, 1))), 0) # 4, 1
                next_enu_location = np.matmul(next_ego2global, next_gt_location) # 3, 1
                next_enu_location = np.concatenate((next_enu_location, np.ones((1, 1))), 0) # 4, 1

                next2cur_gt_center = np.matmul(np.linalg.inv(cur_gt_ego2global), next_enu_location).reshape(-1)[:2] # 3, 1
                offset = next2cur_gt_center - cur_gt_center
                tmp_fut_traj[left_id] = offset
                if vel_tag:
                    gt_vel.append([offset[0]/(left_id+1), offset[1]/(left_id+1)])
                    vel_tag = False
                cur_gt_center = next2cur_gt_center
        
        if vel_tag:
            gt_vel.append([0.0, 0.0])

        tmp_attr_label = np.array((tmp_fut_traj.reshape(-1).tolist() + tmp_fut_mask)).reshape(fut_ts*3)
        gt_attr_label.append(tmp_attr_label.tolist()) # fut_ts*3
    return gt_attr_label, gt_vel

if __name__ == "__main__":
    ## processing GT file
    track_file = '/Users/suhaisheng/Desktop/robosense/global_splits/robosense_global_val.pkl'
    track_data = pkl.load(open(track_file, 'rb'))

    ## processing det file
    # det_file = '/mnt/lustrenew/suhaisheng/robosense_exp/pointpillar_livox_v0.16_clean_results/results.txt'
    det_file = '/Users/suhaisheng/Desktop/backup0628/robosense/robosense_scripts/metrics/sample_results.txt'
    det_data = open(det_file, 'r').readlines()

    seq_dict = defaultdict(list)
    # sort by seq_id
    for i in range(len(track_data)):
        seq_name = track_data[i]['seq_token']
        velodyne_path = track_data[i]['velodyne_path']
        seq_dict[seq_name].append(velodyne_path)
    print(len(list(seq_dict.keys())))

    track_stat = dict()
    for track_line in track_data:
        track_stat[track_line['velodyne_path']] = {
            'name': track_line['annos']['name'], 
            'dimensions': track_line['annos']['dimensions'], 
            'location': track_line['annos']['location'],
            'rotation_y': track_line['annos']['rotation_y'],
            'id': track_line['annos']['id'],
            'gt_attr_label': track_line['gt_attr_label'],
            'ego2global_rotation': track_line['ego2global_rotation'], 
            'ego2global_translation': track_line['ego2global_translation']}

    print('number of det lines:{}'.format(len(det_data)))
    det_stat = dict()
    for det_line in det_data:
        dets = json.loads(det_line.rstrip())
        det_stat[dets['velodyne_path']] = {
            'name': dets['name'], 
            'boxes_lidar': dets['boxes_lidar'], 
            'score': dets['score']}

    missing_count = 0
    fut_mode, fut_ts = 6, 3
    all_metric_results = list()
    for seq_name in tqdm(seq_dict.keys()):
        velo_path_list = sorted(seq_dict[seq_name])
        num_frame = len(velo_path_list)
        # batch-level evaluation
        for frame_id, velo_path in enumerate(velo_path_list):
            ## fetch gt box result
            gt_anno = track_stat[velo_path]

            num_box = len(gt_anno['name'])
            gt_dims = gt_anno['dimensions']  # wlh
            gt_locations = gt_anno['location'] # xyz
            gt_rotation = gt_anno['rotation_y']
            # rotation_y = -(gt_anno['rotation_y'] + np.pi/2)
            gt_track_id = gt_anno['id']
            gt_ego2global_rotation = np.array(gt_anno['ego2global_rotation'])
            gt_ego2global_translation = np.array(gt_anno['ego2global_translation']).reshape(3, 1)
            cur_gt_ego2global = np.concatenate((gt_ego2global_rotation, gt_ego2global_translation), -1)
            cur_gt_ego2global = np.concatenate((cur_gt_ego2global, np.array([0,0,0,1]).reshape(1, 4)), 0)
            gt_label = gt_anno['name']
            gt_bbox = dict()
            gt_bbox['center'] = gt_locations
            gt_bbox['boxes_3d'] = np.concatenate((gt_locations, gt_dims, gt_rotation.reshape(-1, 1)), 1)
            gt_bbox['labels_3d'] = np.array(gt_anno['name'])

            ## fut 3s traj label generation
            gt_attr_label, gt_vel = motion_traj_generate(track_stat,gt_locations,gt_track_id,velo_path_list,frame_id)
            gt_attr_label = gt_anno.get('gt_attr_label', gt_attr_label)
    
            gt_attr_label = torch.from_numpy(np.array(gt_attr_label))
            gt_vel = np.array(gt_vel)
            ## fetch det box result
            if velo_path not in det_stat:
                missing_count += 1
                continue
            det_sample = det_stat[velo_path]
            num_box = len(det_sample['name'])
            if num_box == 0:
                continue
            ## using GT box for evaluation
            # det_labels_3d = gt_bbox['labels_3d']
            # det_scores_3d = np.ones(len(gt_locations))
            # det_boxes_3d = gt_bbox['boxes_3d']

            ## using 3D detection results for evaluation
            det_labels_3d = np.array(det_sample['name'])
            det_scores_3d = np.array(det_sample['score'])
            det_boxes_3d = np.array(det_sample['boxes_lidar'])

            # constant positions
            det_trajs_3d = torch.zeros((len(det_boxes_3d), fut_mode, fut_ts*2))
            # constant velocities
            # det_trajs_3d = torch.ones((len(det_boxes_3d), fut_mode, fut_ts*2))
            # random velocities
            # det_trajs_3d = torch.rand((len(det_boxes_3d), fut_mode, fut_ts*2)) # 0-2

            # filter pred bbox by score_threshold
            score_threshold = 0.6
            bbox_result = dict()
            mask = det_scores_3d > score_threshold
            bbox_result['boxes_3d'] = det_boxes_3d[mask]
            bbox_result['scores_3d'] = det_scores_3d[mask]
            bbox_result['labels_3d'] = det_labels_3d[mask]
            bbox_result['trajs_3d'] = det_trajs_3d[mask]

            matched_bbox_result = assign_pred_to_gt_vip3d(
                bbox_result, gt_bbox, gt_label)

            metric_dict = compute_motion_metric_vip3d(
                gt_bbox, gt_label, gt_attr_label, bbox_result,
                matched_bbox_result)

            all_metric_results.append(metric_dict)

    print(missing_count)
    print(len(all_metric_results))
    evaluate(all_metric_results)
