import IOU_3D
import distance_3D

import numpy as np
import numba
import pickle
import warnings


warnings.filterwarnings('ignore')
label_dict = {
    'Car': 'Car',
    'Pedestrian': 'Pedestrian',
    'Cyclist': 'Cyclist',
    'Bus': 'Bus',
    'Tricyclist': 'Tricyclist',
    'Truck': 'Truck'
}

def d3_box_overlap(boxes, qboxes, criterion=-1):
    boxes = boxes[:, [0, 2, 1, 3, 5, 4, 6]]
    qboxes = qboxes[:, [0, 2, 1, 3, 5, 4, 6]]
    rinc = rotate_iou_gpu_eval(boxes[:, [0, 2, 3, 5, 6]],
                               qboxes[:, [0, 2, 3, 5, 6]], 2)
    d3_box_overlap_kernel(boxes, qboxes, rinc, criterion)
    return rinc


@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
    # ONLY support overlap in CAMERA, not lider.
    N, K = boxes.shape[0], qboxes.shape[0]
    for i in range(N):
        for j in range(K):
            if rinc[i, j] > 0:
                iw = (min(boxes[i, 1], qboxes[j, 1]) - max(
                    boxes[i, 1] - boxes[i, 4], qboxes[j, 1] - qboxes[j, 4]))

                if iw > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = iw * rinc[i, j]
                    if criterion == -1:
                        ua = (area1 + area2 - inc)
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = 1.0
                    rinc[i, j] = inc / ua
                else:
                    rinc[i, j] = 0.0


def gt_to_bboxes_v1(gt_file):
    bboxes_dict = {
        'Car': [],
        'Pedestrian': [],
        'Cyclist': [],
        'Tricyclist': [],
        'Truck': [],
        'Bus': []
    }
    for obj in gt_file:
        cls = label_dict[obj['attribute']]
        bbox = [obj['center'][axis] for axis in ['x', 'y', 'z']] + [
            obj[dim] for dim in ['length', 'width', 'height', 'rotation']
        ]
        bboxes_dict[cls].append(bbox)
    return bboxes_dict


def gt_to_bboxes_v2(gt_file, index):
    bboxes_dict = {
        'Car': [],
        'Pedestrian': [],
        'Cyclist': [],
        'Tricyclist': [],
        'Truck': [],
        'Bus': []
    }
    objs = gt_file
    for i in range(objs['name'].shape[0]):
        cls = label_dict[objs['name'][i]]
        dimensions = np.array(objs['dimensions'][i])[:, [1,0,2]]
        rotation = -(objs['rotation_y'][i] + np.pi/2)

        bbox = objs['location'][i].tolist() + dimensions.tolist() +\
            [rotation]

        bboxes_dict[cls].append(bbox)
    return bboxes_dict

def gt_to_bboxes_drop_outofrange(gt_file, index, point_cloud_range=[0, -46.0, -1, 92, 46.0, 4]):
    bboxes_dict = {
        'Car': [],
        'Pedestrian': [],
        'Cyclist': [],
        'Tricyclist': [],
        'Truck': [],
        'Bus': []
    }
    objs = gt_file
    for i in range(objs['name'].shape[0]):
        cls = label_dict[objs['name'][i]]
        dimensions = np.array(objs['dimensions'][i])
        rotation = objs['rotation_y'][i]
        dimensions = dimensions[[1,0,2]]  # wlh -> lwh
        rotation = -(rotation + np.pi/2)
        location = objs['location'][i]
        location[2] += dimensions[2] / 2. # bottom -> center

        ## use for different range evaluation
        dist = (location[0] ** 2 + location[1] **2) ** 0.5
        if point_cloud_range[3] == 5:
            if dist > 5.0:
                continue
        elif point_cloud_range[3] == 10:
            if dist <=5 or dist > 10:
                continue
        elif point_cloud_range[3] == 30:
            if dist <= 10 or dist > 30:
                continue
        elif location[0]<point_cloud_range[0] or location[0]>point_cloud_range[3] \
                or location[1]<point_cloud_range[1] or location[1]>point_cloud_range[4]:
            continue

        bbox = location.tolist() + dimensions.tolist() + [rotation]
        bboxes_dict[cls].append(bbox)
    return bboxes_dict

def pred_to_bboxes(pred_file, point_cloud_range=None):
    bboxes_dict = {
        'Car': [],
        'Pedestrian': [],
        'Cyclist': [],
        'Tricyclist': [],
        'Truck': [],
        'Bus': []
    }
    for i in range(pred_file['name'].shape[0]):
        cls = label_dict[pred_file['name'][i]]
        bbox = pred_file['location'][i].tolist() + pred_file['dimensions'][i].tolist() +\
            [pred_file['rotation_y'][i]] + [pred_file['score'][i]]
        
        ## Warning: use for different range evaluation
        location = bbox[:3]
        dist = (location[0] ** 2 + location[1] **2) ** 0.5
        if point_cloud_range[3] == 5:
            if dist > 5.0:
                continue
        elif point_cloud_range[3] == 10:
            if dist <=5 or dist > 10:
                continue
        elif point_cloud_range[3] == 30:
            if dist <= 10 or dist > 30:
                continue

        bboxes_dict[cls].append(bbox)
    return bboxes_dict


cls_dict = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist'}


def pred_to_bboxes2(pred_file):
    bboxes_dict = {
        'Car': [],
        'Pedestrian': [],
        'Cyclist': [],
        'Tricyclist': [],
        'Truck': [],
        'Bus': []
    }
    for i, box in enumerate(pred_file):
        cls = cls_dict[box[8]]
        bbox = box[:8]
        bboxes_dict[cls].append(bbox)
    return bboxes_dict


class Eval_result(object):
    def __init__(self):
        self.tp = []
        self.fp = []
        self.fn = []
        self.precision = []
        self.recall = []
        self.AP = 0
        self.F1_score = 0

        self.aos_score = []
        self.aos_half_score = []
        self.ate_score = []
        self.ase_score = []
        self.aoe_score = []
        self.ate = 0  # IOU or distance
        self.ase = 0
        self.aoe = 0
        self.aos = 0
        self.aos_half = 0
        self.max_index = 0
        self.max_F1_score = 0

    def update_results(self, rets):
        tp, fp, fn, aos_score, aos_half_score, ate_score, ase_score, aoe_score = rets
        self.tp.append(tp)
        self.fp.append(fp)
        self.fn.append(fn)
        self.aos_score.append(aos_score)
        self.aos_half_score.append(aos_half_score)
        self.ate_score.append(ate_score)
        self.ase_score.append(ase_score)
        self.aoe_score.append(aoe_score)

    def compute_statistics(self, gt_boxes, pred_boxes, match_type, match_thresh, interpolation_num=40):
        gt_boxes = np.array(gt_boxes)
        pred_boxes = np.array(pred_boxes)
        if gt_boxes.shape[0] == 0 or pred_boxes.shape[0] == 0:
            tp = np.zeros([interpolation_num])
            fp = np.zeros([interpolation_num])
            if pred_boxes.shape[0] == 0:
                fn = len(gt_boxes) * np.ones([interpolation_num])
            else:
                # for j, conf in enumerate(np.linspace(0, 0.95, 20)):
                for j, conf in enumerate(np.linspace(0, 1, interpolation_num, False)):
                    fp[j] = pred_boxes[pred_boxes[:, 7] > conf].shape[0]
                fn = np.zeros([interpolation_num])

            aos_score = np.zeros([interpolation_num])
            aos_half_score = np.zeros([interpolation_num])
            ate_score = np.zeros([interpolation_num])
            ase_score = np.zeros([interpolation_num])
            aoe_score = np.zeros([interpolation_num])
            return [tp, fp, fn, aos_score, aos_half_score, ate_score, ase_score, aoe_score]

        gt_boxes = np.array(gt_boxes)
        pred_boxes = np.array(pred_boxes)
        if match_type == 'IOU':
            #overlap = d3_box_overlap(gt_boxes, pred_boxes).astype(np.float64)
            overlap= IOU_3D.compute_IOU_3d(gt_boxes, pred_boxes)
        elif match_type == 'center_distance':
            overlap = distance_3D.compute_distance_3d(gt_boxes, pred_boxes, 'center')
        elif match_type == 'chp_distance':
            overlap = distance_3D.compute_distance_3d(gt_boxes, pred_boxes, 'chp')
        else:
            raise ValueError('invalid match type:{}'.format(match_type))

        # output:n*m matrix overlap_part[i,j]:3d_iou(gt_boxes[i],dt_boxes[j])
        tp, fp, fn  = np.zeros([interpolation_num]), np.zeros([interpolation_num]), np.zeros([interpolation_num])
        aos_score, aos_half_score = np.zeros([interpolation_num]), np.zeros([interpolation_num])
        ate_score, ase_score, aoe_score = np.zeros([interpolation_num]), np.zeros([interpolation_num]), np.zeros([interpolation_num])
        # for j, conf in enumerate(np.linspace(0, 0.95, 20)):
        for j, conf in enumerate(np.linspace(0, 1, interpolation_num, False)):
            overlap_part = overlap[:, pred_boxes[:, 7] > conf]
            dt_boxes = pred_boxes[pred_boxes[:, 7] > conf]
            if overlap_part.shape[1] == 0:
                tp[j] = 0
                fp[j] = 0
                fn[j] = gt_boxes.shape[0]
                aos_score[j] = 0
                aos_half_score[j] = 0
                # ate_score[j] = 0
                # ase_score[j] = 0
                # aoe_score[j] = 0
                ate_score[j] = 1
                ase_score[j] = 1
                aoe_score[j] = 1
                continue

            matched_list = np.zeros([dt_boxes.shape[0]])
            for i, gt_box in enumerate(gt_boxes):
                num_match = 0
                gt_is_matched = 0
                match_value_list = np.sort(overlap_part[i])
                cur_match_tag = True
                while cur_match_tag and num_match < dt_boxes.shape[0]:
                    # IOU/Distance
                    if match_type == 'IOU':
                        cur_match_value = match_value_list[-1 - num_match]
                        if cur_match_value > match_thresh:
                            cur_match_tag = True
                        else:
                            cur_match_tag = False
                    elif match_type.find('distance') >= 0:
                        cur_match_value = match_value_list[num_match]

                        if cur_match_value < match_thresh:
                            cur_match_tag = True
                        else:
                            cur_match_tag = False
                    else:
                        raise ValueError('invalid match type:{}'.format(match_type))

                    max_match_index = overlap_part[i].tolist().index(cur_match_value)
                    if cur_match_tag and matched_list[max_match_index] == 0:
                        matched_list[max_match_index] = 1
                        tp[j] += 1
                        aos_score[j] += (
                            1.0 + np.cos(gt_boxes[i, 6] - dt_boxes[max_match_index, 6])
                        ) / 2.0
                        # print(gt_boxes[i, 6], dt_boxes[max_match_index, 6], (1.0 + np.cos(gt_boxes[i, 6] - dt_boxes[max_match_index, 6])))
                        aos_half_score[j] += abs(
                            np.cos(
                                gt_boxes[i, 6] - dt_boxes[max_match_index, 6])
                        )
                        tmp_gt, tmp_dt = gt_boxes[i], dt_boxes[max_match_index]

                        if match_type == 'IOU':
                            # ratio, not meter
                            ate_score[j] += distance_3D.compute_distance_3d(
                                tmp_gt.reshape(1, -1), tmp_dt.reshape(1, -1), 'center')[0, 0]
                        elif match_type.find('distance') >= 0:
                            # ratio, not meter
                            ate_score[j] += cur_match_value
                        else:
                            raise ValueError('invalid match type: {}'.format(match_type))
                        aoe_score[j] += abs(gt_boxes[i, 6] - dt_boxes[max_match_index, 6])
                        tmp_dt[:3] = tmp_gt[:3]
                        tmp_dt[6] = tmp_gt[6]   # align the orientation
                        iou3d = IOU_3D.compute_IOU_3d(tmp_gt.reshape(1, -1), tmp_dt.reshape(1, -1))[0, 0]
                        ase_score[j] += 1 - iou3d
                        gt_is_matched = 1
                        break
                    num_match += 1

                if not gt_is_matched:
                    fn[j] += 1
            fp[j] = matched_list.shape[0] - np.sum(matched_list)
        result = [tp, fp, fn, aos_score, aos_half_score, ate_score, ase_score, aoe_score]
        return result

    def compute_pre_rec_map(self):
        fn = np.array(self.fn).sum(axis=0)
        fp = np.array(self.fp).sum(axis=0)
        tp = np.array(self.tp).sum(axis=0) + 1e-7  # avoid divide zero
        aos_score = np.array(self.aos_score).sum(axis=0)
        aos_half_score = np.array(self.aos_half_score).sum(axis=0)
        ase_score = np.array(self.ase_score).sum(axis=0)

        # ase_score = np.array(self.ase_score)
        ate_score = np.array(self.ate_score)
        aoe_score = np.array(self.aoe_score)

        self.precision = tp / (tp + fp)
        self.recall = tp / (tp + fn)
        aos_pre = aos_score / tp
        aos_half_pre = aos_half_score / tp
        ase_pre = ase_score / tp

        cur_aos_score = aos_pre[0]
        cur_aos_half_score = aos_half_pre[0]
        cur_ase_score = ase_pre[0]
        cur_pre = self.precision[0]
        cur_rec = self.recall[0]
        self.AP = 0
        self.aos = 0
        self.aos_half = 0
        self.ate = 0
        self.ase = 0
        self.aoe = 0
        self.F1_score = np.zeros(self.precision.shape[0])
        conf_index = -1
        for i in range(1, self.precision.shape[0]):
            # exclude low recall of noise results
            # if cur_rec < 0.1:
            #     continue
            # if cur_pre >= 0.1:
            self.AP += (cur_rec - self.recall[i]) * cur_pre
            if cur_rec >= 0.1:
                self.aos += (cur_rec - self.recall[i]) * cur_aos_score
                self.aos_half += (cur_rec - self.recall[i]) * cur_aos_half_score
                self.ase += (cur_rec - self.recall[i]) * cur_ase_score
            cur_pre = self.precision[i]
            cur_rec = self.recall[i]
            if cur_pre > 0.8:
                if conf_index == -1:
                    conf_index = i
            cur_aos_score = aos_pre[i]
            cur_aos_half_score = aos_half_pre[i]
            cur_ase_score = ase_pre[i]
        
        # if self.ase == 0:
        #     self.ase = 1.0

        # exclude low recall of noise results
        # if cur_rec >= 0.1:
        #     if cur_pre >= 0.1:
        self.AP += cur_rec * cur_pre
        if cur_rec >= 0.1:
            self.aos += cur_rec * cur_aos_score
            self.aos_half += cur_rec * cur_aos_half_score
            self.ase += cur_rec * cur_ase_score

        # self.aos = np.sum(aos_score)/np.sum(tp)
        # self.aos_half = np.sum(aos_half_score)/np.sum(tp)
        # self.ase = np.sum(ase_score) / np.sum(tp)
        self.ate = np.sum(ate_score) / np.sum(tp)
        self.aoe = np.sum(aoe_score) / np.sum(tp)
        
        self.F1_score = 2 * (
            self.precision * self.recall) / (
            self.precision + self.recall)
        self.max_F1_score = max(self.F1_score)
        self.max_index = self.F1_score.tolist().index(self.max_F1_score)
        # self.max_index = conf_index


def evaluate_results(gt_file, pred_file,
                     point_cloud_range=None,
                     drop_out_range=True,
                     match_type='IOU',
                     thresholds=[0.7,0.5,0.5]):
    # match_type: ['IOU', 'distance', 'size', 'rot']
    # IOU_thresh=[0.7, 0.5, 0.5]
    # dis_thresh=[0.1, 0.1, 0.1]
    match_thresholds = {
        'Car': thresholds[0],
        'Pedestrian': thresholds[1],
        'Cyclist': thresholds[2]
    }
    cls_dict = [
        'Car', 'Pedestrian', 'Cyclist', 'Total'
    ]
    metrics_results = {}
    chp_metrics_results = {}
    for cls in cls_dict:
        metrics_results[cls] = Eval_result()
        chp_metrics_results[cls] = Eval_result()

    pred_dict = dict()
    for pred in pred_file:
        # from ipdb import set_trace; set_trace()
        pred_dict[str(pred['velodyne_path'])] = pred

    for i in range(len(gt_file)):
        
        if i%(len(gt_file)//10) == 0:
            print("current:%d total:%d ratio:%.3f"%(i,len(gt_file),i/len(gt_file)),flush=True)

        # gt_bboxes = gt_to_bboxes_v2(gt_file[i], i)
        # pred_bboxes = pred_to_bboxes(pred_file[i])
        gt_bboxes={}
        if drop_out_range:
            gt_bboxes= gt_to_bboxes_drop_outofrange(gt_file[i][0], i, point_cloud_range)
        else:
            gt_bboxes = gt_to_bboxes_v2(gt_file[i][0], i)

        velodyne_path = gt_file[i][1]
        pred_bboxes = pred_to_bboxes(pred_dict[velodyne_path], point_cloud_range)
        # pred_bboxes = pred_to_bboxes2(pred_file)

        for k, v in match_thresholds.items():  # get tp fp fp ori for each class
            if match_type == 'IOU':
                rets = metrics_results[k].compute_statistics(gt_bboxes[k],
                                                            pred_bboxes[k],
                                                            match_type='IOU',
                                                            match_thresh=v)
                metrics_results[k].update_results(rets)
                metrics_results['Total'].update_results(rets)

            elif match_type == 'distance':
                rets = metrics_results[k].compute_statistics(gt_bboxes[k],
                                                            pred_bboxes[k],
                                                            match_type='center_distance',
                                                            match_thresh=v)
                chp_rets = chp_metrics_results[k].compute_statistics(gt_bboxes[k],
                                                            pred_bboxes[k],
                                                            match_type='chp_distance',
                                                            match_thresh=v)
                                
                metrics_results[k].update_results(rets)
                metrics_results['Total'].update_results(rets)

                chp_metrics_results[k].update_results(chp_rets)
                chp_metrics_results['Total'].update_results(chp_rets)

    for cls in cls_dict:  # get AP Aos for each class
        if np.array(
                metrics_results[cls].tp).sum() == 0:  # remove the invaild cls
            del metrics_results[cls]
            continue
        metrics_results[cls].compute_pre_rec_map()

        if np.array(chp_metrics_results[cls].tp).sum() == 0:  # remove the invaild cls
            del chp_metrics_results[cls]
            continue
        chp_metrics_results[cls].compute_pre_rec_map()

    return metrics_results, chp_metrics_results


if __name__ == '__main__':
    gt_path = "~/gt.pkl"
    pkl_file = open(gt_path, 'rb')
    gt_file = pickle.load(pkl_file)
    pkl_file.close()
    pred_path = '~/det.pkl'
    pred_file = open(pred_path, 'rb')
    pred_file = pickle.load(pred_file)
    pred_file.close()
    metrics_results = evaluate_results(gt_file, pred_file)
    metric_res = {}
    for idx, metrics_result in enumerate(metrics_results):
        if idx == 0:
            print('===== Start Center Point (CP) Eval ======')
        elif idx == 1:
            print('===== Start Closet Hit Point (CHP) Eval ======')
        for k, v in metrics_result.items():  # cls
            try:
                metric_res['3d/{}_AP'.format(k)] = round(metrics_result[k].AP, 3)
                metric_res['3d/{}_F1'.format(k)] = round(metrics_result[k].max_F1_score, 3)

                metric_res['3d/{}_Precision'.format(k)] = round(metrics_result[k].precision[metrics_result[k].max_index], 3)
                metric_res['3d/{}_Recall'.format(k)] = round(metrics_result[k].recall[metrics_result[k].max_index], 3)
                metric_res['3d/{}_AOS'.format(k)] = round(metrics_result[k].aos, 3)
                metric_res['3d/{}_ATE'.format(k)] = round(metrics_result[k].ate, 3)
                metric_res['3d/{}_ASE'.format(k)] = round(metrics_result[k].ase, 3)
                metric_res['3d/{}_AOE'.format(k)] = round(metrics_result[k].aoe, 3)

                log_str = k + ': AP: {}'.format(round(metrics_result[k].AP, 3)) \
                    + ' F1 Score: {}'.format(round(metrics_result[k].max_F1_score, 3)) \
                    + ' Precision: {}'.format(round(metrics_result[k].precision[metrics_result[k].max_index], 3)) \
                    + ' Recall: {}'.format(round(metrics_result[k].recall[metrics_result[k].max_index], 3)) \
                    + ' Threshold: {}'.format(round((metrics_result[k].max_index) * 0.05, 3)) \
                    + ' AOS: {}%'.format(round(metrics_result[k].aos * 100, 3)) \
                    + ' AOS@0.5: {}%'.format(round(metrics_result[k].aos_half * 100, 3)) \
                    + ' ATE: {}%'.format(round(metrics_result[k].ate * 100, 3)) \
                    + ' ASE: {}'.format(round(metrics_result[k].ase, 3)) \
                    + ' AOE [rad]/[degree]: {}/{}'.format(
                        round(metrics_result[k].aoe, 3),
                        round(metrics_result[k].aoe / np.pi * 180, 3))

                print(log_str)
            except:
                pass
        if idx == 0:
            print('===== End Center Point (CP) Eval ======')
        elif idx == 1:
            print('===== End Closet Hit Point (CHP) Eval ======')

    print('end eval')
    print('testing')
