import numpy as np
import math

def compute_point_of_intersection(vector1,vector2):
    #vector:[x1,y1,x2,y2]
    x1=vector1[0]
    y1=vector1[1]
    x2=vector1[2]
    y2=vector1[3]
    x3=vector2[0]
    y3=vector2[1]
    x4=vector2[2]
    y4=vector2[3]
    if ((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))==0 or ((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))==0:
        return [0,0],False #two parallel lines
    else:
        x0=((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
        y0=((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
        if (x0>=max(min(x1,x2),min(x3,x4)) and x0<=min(max(x1,x2),max(x3,x4)) #whether the point is on the lines
            and y0>=max(min(y1,y2),min(y3,y4)) and y0<=min(max(y1,y2),max(y3,y4))):
            return [x0,y0],True
        else:
            return [x0,y0],False

def point_in_quadrilateral(p, corners):
    #p: point [x,y]
    #corners: 8 points of rect [x1,y1,x2,y2,x3,y3,x4,y4]
    ab0 = corners[1,0] - corners[0,0]
    ab1 = corners[1,1] - corners[0,1]

    ad0 = corners[3,0] - corners[0,0]
    ad1 = corners[3,1] - corners[0,1]

    ap0 = p[0] - corners[0,0]
    ap1 = p[1] - corners[0,1]

    abab = ab0 * ab0 + ab1 * ab1 #calculate the length of ab
    abap = ab0 * ap0 + ab1 * ap1 #calculate the length of ap'
    adad = ad0 * ad0 + ad1 * ad1 #calculate the length of ad
    adap = ad0 * ap0 + ad1 * ap1 #calculate the length of ap''

    return abab+1e-5 >= abap and abap+1e-5 >= 0 and adad+1e-5 >= adap and adap+1e-5 >= 0

def sort_vertex_in_convex_polygon(int_pts):
    def cmp(pt, center):
        vx = pt[0] - center[0]
        vy = pt[1] - center[1]
        d = math.sqrt(vx * vx + vy * vy)
        vx /= d
        vy /= d
        if vy < 0:
            vx = -2 - vx
        return vx

    num_of_inter=len(int_pts)
    if num_of_inter > 0:
        center = [0, 0]
        for i in range(num_of_inter):
            center[0] += int_pts[i][0]
            center[1] += int_pts[i][1]
        center[0] /= num_of_inter
        center[1] /= num_of_inter # get the average position of points
        int_pts.sort(key=lambda x: cmp(x, center))

def compute_inter_area(int_pts):
    def trangle_area(a, b, c):
        return ((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) *
                (b[0] - c[0])) / 2.0

    area_val = 0.0
    for i in range(len(int_pts) - 2):
        area_val += abs(
            trangle_area(int_pts[0], int_pts[i + 1],
                          int_pts[i + 2]))
    return area_val


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

def compute_IOU_3d(bbox1,bbox2):
    IOU_matrix=np.zeros([bbox1.shape[0],bbox2.shape[0]])
    for row in range(bbox1.shape[0]):
        for col in range(bbox2.shape[0]):
            inter_height=min(bbox1[row,2]+0.5*bbox1[row,5],bbox2[col,2]+0.5*bbox2[col,5])-max(bbox1[row,2]-0.5*bbox1[row,5],bbox2[col,2]-0.5*bbox2[col,5])#calculate overlap on height
            if inter_height<=0:
                IOU_matrix[row,col]=0
                continue
            dis=math.sqrt((bbox1[row,0]-bbox2[col,0])**2+(bbox1[row,1]-bbox2[col,1])**2)
            if dis>(bbox1[row,3]+bbox1[row,4]+bbox2[col,3]+bbox2[col,4]):
                IOU_matrix[row,col]=0
                continue
            corners_1=compute_box_3d(bbox1[row])
            corners_2=compute_box_3d(bbox2[col])
            polygon_points=[]
            for point in corners_1[:4]:
                in_rect=point_in_quadrilateral(point[:2],corners_2[:4,:2])
                if in_rect:
                    polygon_points.append(point[:2])
                    #print(point)
            for point in corners_2[:4]:
                in_rect=point_in_quadrilateral(point[:2],corners_1[:4,:2])
                if in_rect:
                    polygon_points.append(point[:2])
                    #print(point)
            for i,line1 in enumerate(corners_1[:4]):
                for j,line2 in enumerate(corners_1[:4]):
                    v1=[corners_1[(i+1)%4][0],corners_1[(i+1)%4][1],corners_1[(i)%4][0],corners_1[(i)%4][1]] #get vector1 (edge) from rectangle1
                    v2=[corners_2[(j+1)%4][0],corners_2[(j+1)%4][1],corners_2[(j)%4][0],corners_2[(j)%4][1]] #get vector2 (edge) from rectangle2
                    point,is_intersect=compute_point_of_intersection(v1,v2)
                    if is_intersect:
                        polygon_points.append(point)
                        #print(point)
            sort_vertex_in_convex_polygon(polygon_points)#sorted clockwise
            inter_area=compute_inter_area(polygon_points)
            inter_volume=inter_area*inter_height
            volume_1=bbox1[row,3]*bbox1[row,4]*bbox1[row,5]
            volume_2=bbox2[col,3]*bbox2[col,4]*bbox2[col,5]
            total_volume=volume_1+volume_2-inter_volume
            IOU_3d=inter_volume/total_volume
            IOU_matrix[row,col]=IOU_3d
    return IOU_matrix

if __name__ == '__main__':
    bbox1=np.array([[16,1,2,2,2,4,0.7854,6],[6,1,2,2,2,4,0,1],[6,11,2,4,2,4,0,3],
                     [16,1,2,2,2,4,0,5],[16,11,2,2,2,4,0,7],[26,1,2,2,2,4,0,9],[26,11,2,4,2,4,0,11],[36,1,2,2,2,4,0,13],[-16,1,2,2,2,4,0,15]
                    ])
    bbox2=np.array([[16,1,2,2,2,4,0.7854,6],[6,1,1,2,2,2,0,2],[6,11,2,4,2,4,1.5708,4],[16,12,2,2,2,4,0,8],
                     [26,2,1,2,2,2,0,10],[26,11,2,4,2,4,0.7854,12],
                     [36,2,2,2,2,4,0.7854,14]])
    IOU_3d=compute_IOU_3d(bbox1,bbox2)
    print(IOU_3d)
