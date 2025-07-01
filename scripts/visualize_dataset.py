import numpy as np
import os
import cv2
import pickle as pkl
import matplotlib.pyplot as plt
from collections import defaultdict
import copy
import random, string


def rotz(t):
    """ Rotation about the z-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

class Box_3d(object):
    """ 3d object label """

    def __init__(self, results):
        self.h = results[5]  # box height
        self.w = results[4]   # box width
        self.l = results[3]   # box length (in meters)
        self.t = results[:3]  # location (x,y,z) in camera coord.
        self.ry = -(results[6] + np.pi / 2)   # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

def compute_box_3d(bbox):
    obj = Box_3d(bbox)
    # compute rotational matrix around yaw axis
    R = rotz(obj.ry)

    # 3d bounding box dimensions
    l = obj.l
    w = obj.w
    h = obj.h

    """
        4 -------- 5
       /|         /|
      7 -------- 6 .
      | |        | |
      . 0 -------- 1
      |/         |/
      3 -------- 2
    """
    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    # z_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
    z_corners = [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2]
    t_corners = [1, 1, 1, 1, 1, 1, 1, 1]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + obj.t[0]
    corners_3d[1, :] = corners_3d[1, :] + obj.t[1]
    corners_3d[2, :] = corners_3d[2, :] + obj.t[2]
    corners_3d = np.vstack([corners_3d, t_corners])
    return np.transpose(corners_3d)

def HS2Img(coor_hs, hs2ego, ego2cam, cam_K, cam_dist=None, mode="PH"):
    '''
    project from hs coor to pixel coor
    mode: OV or PH
    '''
    cam_K = copy.deepcopy(cam_K)
    coor_ego = np.dot(hs2ego, np.transpose(coor_hs))
    coor_cam = np.dot(ego2cam, coor_ego)[:3]
    if mode == "PH":
        coor_pixel = np.dot(cam_K, coor_cam)/coor_cam[2]
        coor_pixel = np.transpose(coor_pixel)[:,:2].astype(np.int32)
    elif mode == "OV":
        tmp = copy.deepcopy(cam_K[0][0:2])
        cam_K[0][0:2] = cam_K[1][0:2]
        cam_K[1][0:2] = tmp

        coor_cam_tmp = copy.deepcopy(coor_cam)
        invpol = cam_dist[1]
        coor_cam_tmp = np.transpose(coor_cam_tmp)
        norm = (coor_cam_tmp[:, 0] ** 2 + coor_cam_tmp[:, 1] ** 2) ** 0.5
        theta = np.arctan(coor_cam_tmp[:, 2] / norm)
        rho = 0
        for i in range(len(invpol)):
            rho += invpol[i] * theta ** i
        x = coor_cam_tmp[:, 0] / norm * rho
        y = coor_cam_tmp[:, 1] / norm * rho

        coor_cam_tmp[:, 0] = x
        coor_cam_tmp[:, 1] = y
        coor_cam_tmp[:, 2] = 1
        coor_pixel = np.dot(cam_K, np.transpose(coor_cam_tmp))
        coor_pixel = np.transpose(coor_pixel)[:,:2].astype(np.int32)
        coor_cam[1] = -coor_cam[1]
        coor_cam[2] = -coor_cam[2]
    return coor_pixel, coor_cam

def draw_lines(image, cam_name, box_corner_pixel, color):
    lines = [[0,1],[1,2],[2,3],[3,0],
            [0,4],[1,5],[2,6],[3,7],
            [4,5],[5,6],[6,7],[7,4]]
    for k, line in enumerate(lines):
        cv2.line(image, tuple(box_corner_pixel[line[0]]), tuple(box_corner_pixel[line[1]]), color, 2, 2)
    return image


if __name__ == "__main__":
    robosense_data_path = '~/robosense_dataset/image_trainval'
    robosense_seq_data = '~/robosense_seq'
    pkl_file = '~/robosense_local_val.pkl'
    pkl_data = pkl.load(open(pkl_file, 'rb'))
    draw_bbox = True
    mode = "PH" # "PH", "OV"

    seq_dict = defaultdict(list)
    ts_dict = defaultdict(list)
    scene_dict = defaultdict(list)
    for sample_index in range(len(pkl_data)):
        cam_front_ov = pkl_data[sample_index]['images']['cams']['CAM_FRONT_OV']
        timestamp = cam_front_ov['timestamp']
        seq_token = pkl_data[sample_index]['seq_token']
        seq_dict[seq_token].append({'timestamp': timestamp, 'data': pkl_data[sample_index]})
        ts_dict[seq_token].append(timestamp)
        scene_dict[seq_token] = pkl_data[sample_index]['map_token']

    seq_token_list = list(seq_dict.keys())
    target_seq = random.choice(seq_token_list)
    print('random select seq id:{} for visualization...'.format(target_seq))
    for seq_id, seq_item in seq_dict.items():
        if seq_id != target_seq:
            continue
        seq_dir = os.path.join(robosense_seq_data, str(seq_id))
        if not os.path.exists(seq_dir):
            os.makedirs(seq_dir)

        sorted_ts_list = sorted(ts_dict[seq_id])
        sorted_seq_dict_list = list()
        for ts in sorted_ts_list:
            ts_index = ts_dict[seq_id].index(ts)
            sorted_seq_dict_list.append(seq_dict[seq_id][ts_index])

        start_box_id = 1
        box_id_list = list()
        for frame_id in range(len(sorted_seq_dict_list)):
            timestamp = sorted_seq_dict_list[frame_id]['timestamp']
            data = sorted_seq_dict_list[frame_id]['data']
            cam_front_ov = data['images']['cams']['CAM_FRONT_OV']
            cam_left_ov = data['images']['cams']['CAM_LEFT_OV']
            cam_right_ov = data['images']['cams']['CAM_RIGHT_OV']
            cam_back_ov = data['images']['cams']['CAM_BACK_OV']
            cam_front = data['images']['cams']['CAM_FRONT']
            cam_left = data['images']['cams']['CAM_LEFT']
            cam_right = data['images']['cams']['CAM_RIGHT']
            cam_back = data['images']['cams']['CAM_BACK']
            cam_front_ov_data_path = robosense_data_path + cam_front_ov['data_path']
            cam_left_ov_data_path = robosense_data_path + cam_left_ov['data_path']
            cam_right_ov_data_path = robosense_data_path + cam_right_ov['data_path']
            cam_back_ov_data_path = robosense_data_path + cam_back_ov['data_path']
            cam_front_data_path = robosense_data_path + cam_front['data_path']
            cam_left_data_path = robosense_data_path + cam_left['data_path']
            cam_right_data_path = robosense_data_path + cam_right['data_path']
            cam_back_data_path = robosense_data_path + cam_back['data_path']
            print('start loading images..')
            if mode == "OV":
                cam_front_ov_data = cv2.imread(cam_front_ov_data_path)  
                cam_left_ov_data = cv2.imread(cam_left_ov_data_path)   
                cam_right_ov_data = cv2.imread(cam_right_ov_data_path) 
                cam_back_ov_data = cv2.imread(cam_back_ov_data_path)  
                
                cam_name_data_map = {
                    'CAM_FRONT_OV': cam_front_ov_data,
                    'CAM_RIGHT_OV':cam_right_ov_data,
                    'CAM_LEFT_OV': cam_left_ov_data,
                    'CAM_BACK_OV': cam_back_ov_data
                }
            elif mode == "PH":
                cam_front_data = cv2.imread(cam_front_data_path) 
                cam_left_data = cv2.imread(cam_left_data_path) 
                cam_right_data = cv2.imread(cam_right_data_path) 
                cam_back_data = cv2.imread(cam_back_data_path)

                cam_name_data_map = {
                    'CAM_FRONT': cam_front_data,
                    'CAM_LEFT': cam_left_data,
                    'CAM_RIGHT': cam_right_data,
                    'CAM_BACK': cam_back_data
                }
            # GT: sensor2lidar transform
            # plot 2D bbox
            print(seq_id, frame_id, timestamp, scene_dict[seq_id])
            
            if mode == "OV":
                cam_name_list = ['CAM_FRONT_OV', 'CAM_LEFT_OV', 'CAM_RIGHT_OV', 'CAM_BACK_OV']
                image_shape = cam_front_ov_data.shape
            elif mode == 'PH':
                cam_name_list = ['CAM_FRONT', 'CAM_LEFT', 'CAM_RIGHT', 'CAM_BACK']
                image_shape = cam_front_data.shape

            image_list = list()
            json_dict = dict()
            det_rst = list()
            name = data['annos']['name']
            dimensions = data['annos']['dimensions']
            location = data['annos']['location']
            rotation_y = data['annos']['rotation_y']
            cls_mapping = {'Pedestrian': "1", 'Car': "2", 'Cyclist': "3"}
            for box_id in range(len(name)):
                box_3d_id = box_id
                cls_name = name[box_id]
                if cls_name == 'Car':
                    color = (0, 0, 255)
                elif cls_name == 'Pedestrian':
                    color = (0, 255, 0)
                elif cls_name == 'Cyclist':
                    color = (255, 0, 0)
                w, l, h = dimensions[box_id]
                x, y, z = location[box_id]
                z += h /2 
                yaw = rotation_y[box_id]
                bbox = [x,y,z,l,w,h,yaw]
                box_corner = compute_box_3d(np.array(bbox, dtype='float64'))
                bid = ''.join(random.choices(string.ascii_uppercase+string.ascii_lowercase+string.digits,k=8))
                while bid in box_id_list:
                    bid = ''.join(random.choices(string.ascii_uppercase+string.ascii_lowercase+string.digits,k=8))
                box_id_list.append(bid)
                # print(box_id, start_box_id, start_box_id+box_id)
                for cam_name in cam_name_list:
                    cam_K = data['images']['cams'][cam_name]['cam_intrinsic']
                    cam_dist = data['images']['cams'][cam_name]['cam_dist']
                    sensor2ego_translation = data['images']['cams'][cam_name]['sensor2ego_translation']
                    sensor2ego_rotation = data['images']['cams'][cam_name]['sensor2ego_rotation']
                    cam2ego = np.hstack((sensor2ego_rotation, sensor2ego_translation.reshape(-1, 1)))
                    cam2ego = np.vstack((cam2ego, np.array([0,0,0,1]).reshape(1, 4)))
                    ego2cam = np.linalg.inv(cam2ego)

                    hs2ego = np.eye(4)
                    box_corner_pixel, box_corner_cam = HS2Img(box_corner, hs2ego, ego2cam, cam_K, cam_dist, mode)
                    # box_corner_pixel_mean = np.mean(box_corner_pixel, axis = 0)
                    min_x, min_y = min(box_corner_pixel[:,0]), min(box_corner_pixel[:,1])
                    max_x, max_y = max(box_corner_pixel[:,0]), max(box_corner_pixel[:,1])
                    # if min_x < 0 or max_x >= image_shape[1] or min_y < 0 or max_y >= image_shape[0]:
                    #     continue
                    if min_x >= image_shape[1] or min_y >= image_shape[0] or max_x <= 0 or max_y <= 0:
                        continue
                    min_x = max(0, min_x)
                    max_x = min(image_shape[1], max_x)
                    min_y = max(0, min_y)
                    max_y = min(image_shape[0], max_y)
                    if np.mean(box_corner_cam, axis = 1)[2] < -0.5:
                        continue
                    box_corner_pixel = box_corner_pixel.tolist()

                    cam_color = (0, 0, 0)
                    def draw_pic(image, cam_name, box_corner_pixel, color, box_id, type='2d'):
                        if type == '3d':
                            # draw 3d box
                            image = draw_lines(image, cam_name, box_corner_pixel, color)
                            image = cv2.putText(image, '{}'.format(box_id), (box_corner_pixel[2][0], box_corner_pixel[2][1]+4), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2) 
                        elif type == '2d':
                            ## draw 2d box
                            image = cv2.rectangle(image, (min_x, min_y), (max_x, max_y), color, 2)
                            image = cv2.putText(image, '{}'.format(box_id), (max_x, max_y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1) 
                        image = cv2.putText(image, '{}*{}'.format((max_x - min_x), (max_y - min_y)), (max_x, max_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3) 
                        return image

                    if draw_bbox:
                        if cam_name not in cam_name_data_map:
                            raise ValueError('invalid cam name:{}'.format(cam_name))
                        cam_name_data_map[cam_name] = draw_pic(cam_name_data_map[cam_name], cam_name, box_corner_pixel, color, box_id+start_box_id)

            start_box_id += len(name)
            # fisheye: (720, 1280, 3)
            # pinhole: (1080, 1920, 3)
            if mode == 'OV':
                cam_front_ov_data = cv2.putText(cam_front_ov_data, 'FRONT', (cam_front_ov_data.shape[1]-300, cam_front_ov_data.shape[0]-100), cv2.FONT_HERSHEY_COMPLEX, 2, cam_color, 3)
                cam_right_ov_data = cv2.putText(cam_right_ov_data, 'RIGHT', (cam_right_ov_data.shape[1]-300, cam_right_ov_data.shape[0]-100), cv2.FONT_HERSHEY_COMPLEX, 2, cam_color, 3)
                cam_left_ov_data = cv2.putText(cam_left_ov_data, 'LEFT', (cam_left_ov_data.shape[1]-300, cam_left_ov_data.shape[0]-100), cv2.FONT_HERSHEY_COMPLEX, 2, cam_color, 3) 
                cam_back_ov_data = cv2.putText(cam_back_ov_data, 'BACK', (cam_back_ov_data.shape[1]-300, cam_back_ov_data.shape[0]-100), cv2.FONT_HERSHEY_COMPLEX, 2, cam_color, 3) 
                
                cam_nv_data_1 = np.hstack((cam_front_ov_data, cam_right_ov_data))
                cam_nv_data_2 = np.hstack((cam_left_ov_data, cam_back_ov_data))
            elif mode == 'PH':
                cam_front_data = cv2.putText(cam_front_data, 'FRONT', (cam_front_data.shape[1]-300, cam_front_data.shape[0]-100), cv2.FONT_HERSHEY_COMPLEX, 2, cam_color, 3) 
                cam_left_data = cv2.putText(cam_left_data, 'LEFT', (cam_left_data.shape[1]-300, cam_left_data.shape[0]-100), cv2.FONT_HERSHEY_COMPLEX, 2, cam_color, 3) 
                cam_right_data = cv2.putText(cam_right_data, 'RIGHT', (cam_right_data.shape[1]-300, cam_right_data.shape[0]-100), cv2.FONT_HERSHEY_COMPLEX, 2, cam_color, 3)   
                cam_back_data = cv2.putText(cam_back_data, 'BACK', (cam_back_data.shape[1]-300, cam_back_data.shape[0]-100), cv2.FONT_HERSHEY_COMPLEX, 2, cam_color, 3) 
                
                cam_nv_data_1 = np.hstack((cam_front_data, cam_right_data))
                cam_nv_data_2 = np.hstack((cam_left_data, cam_back_data))
            cam_nv_data = np.vstack((cam_nv_data_1, cam_nv_data_2))

            cv2.imwrite(os.path.join(seq_dir, '{:06d}_{}_{}_{:06d}.png'.format(seq_id, scene_dict[seq_id], mode, frame_id)), cam_nv_data)

            # plt.figure('image', figsize=(10, 5))
            # plt.imshow(cam_nv_data[:,:,::-1])
            # plt.show()
        break
            