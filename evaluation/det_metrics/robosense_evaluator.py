# Standard Library
import builtins
import json
import copy
import numpy as np
import pickle
# Import from third library
import time

# fix pycocotools py2-style bug
builtins.unicode = str


class RoboSenseEvaluator():
    def __init__(self, gt_file, point_cloud_range=[-44.8, -44.8, -1, 44.8, 44.8, 4],
                 drop_out_range=True,
                 match_type='distance',
                 thresholds=[0.05,0.05,0.05]):
        """
        Arguments:
            gt_file (str): directory or json file of annotations
            iou_types (str): list of iou types of [keypoints, bbox, segm]
        """
        self.gt_file = gt_file
        self.drop_out_range = drop_out_range
        self.point_cloud_range = point_cloud_range
        self.match_type = match_type
        self.thresholds = thresholds

    def load_dts(self, res_file, res):
        out = []
        if res is not None:
            out = res
        else:
            print(f'loading res from {res_file}')
            out = []
            with open(res_file, 'r') as f:
                for line in f:
                    out.append(json.loads(line))

        # out = [x for res_gpus in out for res_bs in res_gpus for x in res_bs]
        for idx in range(len(out)):
            for k, v in out[idx].items():
                if isinstance(v, list):
                    out[idx][k] = np.array(v)

        return out

    def eval(self, res_file, class_names, robosense_infos, res, **kwargs):
        if 'annos' not in robosense_infos[0].keys():
            return None, {}
        det_annos = self.load_dts(res_file, res)
        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [(copy.deepcopy(info['annos']), info['velodyne_path']) for info in robosense_infos]

        from robosense_object_eval import evaluate_results
        print(self.match_type)
        metrics_results = evaluate_results(eval_gt_annos, eval_det_annos,
                                        point_cloud_range=self.point_cloud_range,
                                        drop_out_range=self.drop_out_range,
                                        match_type=self.match_type,
                                        thresholds=self.thresholds)
        metric_res = {}
        # type: dict(cls: metric) -> type_cls_metric: value
        # metric_list = ['AP', 'F1', 'Precision', 'Recall', 'Aos', 'IOU']
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

        return metrics_results



if __name__ == "__main__":
   
    # gt_file = '~/gt.pkl'
    # res_file = '~/det.pkl'
    gt_file = '/Users/suhaisheng/Desktop/backup0628/robosense/global_splits/robosense_local_val.pkl'
    res_file = '/Users/suhaisheng/Desktop/backup0628/robosense/robosense_scripts/evaluation/sample_result.txt'
    evaluator = RoboSenseEvaluator(gt_file)

    with open(gt_file, 'rb') as f:
        robosense_infos = pickle.load(f)

    class_names = ['Car', 'Pedestrian', 'Cyclist']

    det_annos = open(res_file, 'r').readlines()

    converted_det_annos = list()
    for det_line in det_annos:
        tmp_dict = dict()
        dets = json.loads(det_line.rstrip())
        for key in dets.keys():
            tmp = np.array(dets[key])
            tmp_dict[key] = tmp
        converted_det_annos.append(tmp_dict)
    metrics_result = evaluator.eval(res_file, class_names, robosense_infos, converted_det_annos)
