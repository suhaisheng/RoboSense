import numpy as np
import math


class Box_3d(object):
    """ 3d object label """
    def __init__(self, results):
        self.h = results[5]  # box height
        self.w = results[4]   # box width
        self.l = results[3]   # box length (in meters)
        self.t = results[:3]  # location (x,y,z) in camera coord.
        self.ry =results[6]   # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

def rotz(t):
    """ Rotation about the z-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def compute_box_3d(bbox):
    obj=Box_3d(bbox)
    # compute rotational matrix around yaw axis
    R = rotz(obj.ry)

    # 3d bounding box dimensions
    l = obj.l
    w = obj.w
    h = obj.h

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + obj.t[0]
    corners_3d[1, :] = corners_3d[1, :] + obj.t[1]
    corners_3d[2, :] = corners_3d[2, :] + obj.t[2]
    return np.transpose(corners_3d)

def compute_distance_3d(bbox1, bbox2, dis_type=None):  # gt_boxes, pred_boxes
    # [x, y, z, l, w, h, alpha]
    # return: CHP dis mat, center dis mat
    dis_mat = np.zeros((bbox1.shape[0], bbox2.shape[0]))
    for row in range(bbox1.shape[0]):
        for col in range(bbox2.shape[0]):
            if dis_type == 'center':
                gt_dis = math.sqrt((bbox1[row, 0]**2 + bbox1[row, 1]**2))    
                # pred_dis = math.sqrt((bbox2[col, 0]**2 + bbox2[col, 1]**2))
                pred_dis = math.sqrt(((bbox2[col, 0] - bbox1[row, 0])**2 + (bbox2[col, 1] - bbox1[row, 1])**2))
                center_rel_dis = pred_dis / gt_dis  # CDR
                # center_rel_dis = pred_dis  # CD
                dis_mat[row, col] = center_rel_dis  # can be larger than 1
            elif dis_type == 'chp':
                gt_corners = compute_box_3d(bbox1[row])
                gt_dis = gt_corners[:, 0]**2 + gt_corners[:, 1]**2
                min_gt_dis = math.sqrt(min(gt_dis))
                min_gt_ind = np.argmin(gt_dis)
                gt_chp = gt_corners[min_gt_ind]

                pred_corners = compute_box_3d(bbox2[col])
                pred_dis = pred_corners[:, 0]**2 + pred_corners[:, 1]**2
                min_pred_dis = math.sqrt(min(pred_dis))
                min_pred_ind = np.argmin(pred_dis)
                pred_chp = pred_corners[min_pred_ind]

                # chp_rel_dis = abs(min_pred_dis - min_gt_dis) / min_gt_dis
                chp_dis = math.sqrt((gt_chp[0] - pred_chp[0]) ** 2 \
                    + (gt_chp[1] - pred_chp[1]) ** 2 \
                    + (gt_chp[2] - pred_chp[2]) **2)
                chp_rel_dis = chp_dis / min_gt_dis
                dis_mat[row, col] = chp_rel_dis  # can be larger than 1
            else:
                raise ValueError('invalid dis type:{}'.format(dis_type))

    return dis_mat


if __name__ == '__main__':
    bbox1 = np.array([[16,1,2,2,2,4,0.7854,6],[6,1,2,2,2,4,0,1],[6,11,2,4,2,4,0,3],
                     [16,1,2,2,2,4,0,5],[16,11,2,2,2,4,0,7],[26,1,2,2,2,4,0,9],[26,11,2,4,2,4,0,11],[36,1,2,2,2,4,0,13],[-16,1,2,2,2,4,0,15]
                    ])
    bbox2 = np.array([[16,1,2,2,2,4,0.7854,6],[6,1,1,2,2,2,0,2],[6,11,2,4,2,4,1.5708,4],[16,12,2,2,2,4,0,8],
                     [26,2,1,2,2,2,0,10],[26,11,2,4,2,4,0.7854,12],
                     [36,2,2,2,2,4,0.7854,14]])
    IOU_3d = compute_distance_3d(bbox1, bbox2)
    print(IOU_3d)
