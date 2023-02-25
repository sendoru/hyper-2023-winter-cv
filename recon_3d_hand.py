import time
import cv2
import mediapipe as mp
import open3d as o3d 
import numpy as np
import copy
from constants import *

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
cap_l = cv2.VideoCapture("./output_L_near.mp4")
cap_r = cv2.VideoCapture("./output_R_near.mp4")

def map_hand_to_2d(hands, img_width: int, img_heigth: int):
    ret = []
    if type(hands.multi_hand_landmarks) == type(None):
        return np.array([], dtype=np.float64)
    for hand_no, hand_landmarks in enumerate(hands.multi_hand_landmarks):
        ret.append(np.zeros((21, 2), dtype=np.float64))
        for i in range(21): # 0~20
            x = hand_landmarks.landmark[mp_hands.HandLandmark(i).value].x * img_width
            y = hand_landmarks.landmark[mp_hands.HandLandmark(i).value].y * img_heigth
            z = hand_landmarks.landmark[mp_hands.HandLandmark(i).value].z
            ret[-1][i][0] = x
            ret[-1][i][1] = y
    return ret

# 이거 일단 양쪽 카메라에서 손이 1개씩만 있는 경우를 가정하고 해야겠다
# TODO disortion 보정한 intrinsic matrix 적용
def recon_3d_hand_points(img_l: np.ndarray, img_r: np.ndarray,
                    hands_l, hands_r,
                    l_int: np.ndarray=L_INTRINSIC, r_int:np.ndarray=R_INTRINSIC,
                    l_ext: np.ndarray=L_EXTRINSIC, r_ext: np.ndarray=R_EXTRINSIC):
    if img_l.shape != img_r.shape:
        raise ValueError
    img_width = img_l.shape[1]
    img_height = img_l.shape[0]

    hand_l_xy = map_hand_to_2d(hands_l, img_width, img_height)
    hand_r_xy = map_hand_to_2d(hands_r, img_width, img_height)

    if not (len(hand_l_xy) == 1 and len(hand_r_xy) == 1):
        return None
    ret = cv2.triangulatePoints(l_int @ l_ext, r_int @ r_ext, hand_l_xy[0].T, hand_r_xy[0].T)

    # homogenous(x, y, z, k) -> (x/k, y/k, z/k)
    ret = ret[:3,:] / ret[3,:]
    return ret

def connect_3d_hand_points(points: np.ndarray):
    if type(points) == type(None):
        return None
    if points.shape != (3, 21):
        raise ValueError("array size must be 3 x 21. Try using transpose matrix if the size is 21 x 3.")
    points = points.T
    points[:,1] *= -1
    points_o3d = o3d.utility.Vector3dVector(points)
    pointcloud = o3d.geometry.PointCloud(points=points_o3d)

    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(points)
    lines.lines = o3d.utility.Vector2iVector(list(mp_hands.HAND_CONNECTIONS))
    
    return pointcloud, lines

def recon_3d_hand(img_l: np.ndarray, img_r: np.ndarray,
                    hands_l, hands_r,
                    l_int: np.ndarray=L_INTRINSIC, r_int:np.ndarray=R_INTRINSIC,
                    l_ext: np.ndarray=L_EXTRINSIC, r_ext: np.ndarray=R_EXTRINSIC):
    points = recon_3d_hand_points(img_l, img_r, hands_l, hands_r, l_int, r_int, l_ext, r_ext)
    return connect_3d_hand_points(points)

