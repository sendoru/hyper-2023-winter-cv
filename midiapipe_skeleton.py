import time
import cv2
import mediapipe as mp
import open3d as o3d 
import numpy as np
import scipy as sp
import copy
from collections import deque

from constants import *
from recon_3d_hand import recon_3d_hand

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
cap_l = cv2.VideoCapture("./output_L_near.mp4")
cap_r = cv2.VideoCapture("./output_R_near.mp4")
image_width = 640
image_height = 480

class LineWithTiming:
    def __init__(self, t_prev: int, p_prev: np.ndarray, t_cur: int, p_cur: np.ndarray):
        self.t_prev = t_prev
        self.p_prev = p_prev
        self.t_cur = t_cur
        self.p_cur = p_cur

        self.pointcloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array([p_prev, p_cur])))
        self.lines = o3d.geometry.LineSet()
        self.lines.points = self.pointcloud.points
        self.lines.lines = o3d.utility.Vector2iVector([[0, 1]])

    def get_p_prev(self):
        return self.pointcloud.points[0]

    def get_p_cur(self):
        return self.pointcloud.points[-1]

def preprocess(image: np.ndarray, intrinsic_mat: np.ndarray, distortion_coeff: np.ndarray, hands):
    image = cv2.undistort(image, L_INTRINSIC, L_DISTORTION)
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return results

def draw_hand(results, image):
    for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
        print(f'HAND NUMBER: {hand_no+1}')
        
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())


# initialize
vis = o3d.visualization.Visualizer()
vis.create_window(width=1600, height=900)
threshold = 0.01
icp_iteration = 100
save_image = False

hands_l = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

hands_r = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

recent_points = deque()
recent_timings = deque()
prev_point = None
frame_no = 0


# loop
while cap_l.isOpened():
    time_elapsed = time.time()
    success_l, image_l = cap_l.read()
    success_r, image_r = cap_r.read()
    if not (success_l and success_r):
        print("Ignoring empty camera frame.")
    # If loading a video, use 'break' instead of 'continue'.
        break
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    results_l = preprocess(image_l, L_INTRINSIC, L_DISTORTION, hands_l)
    results_r = preprocess(image_r, R_INTRINSIC, R_DISTORTION, hands_r)
    if results_l.multi_hand_landmarks:
        draw_hand(results_l, image_l)
    if results_r.multi_hand_landmarks:
        draw_hand(results_r, image_r)

    hand_3d = recon_3d_hand(image_l, image_r, results_l, results_r)

    # 해볼만한거
    # 1. 범위에 제약조건을 걸자
    # 현실적으로 범위가 4m? 정도 이상 나올일이 없을 것 같으니까 그냥 그 밖으로 튀어나가면 점을 렌더링하지 않음
    # 4m도 너무크네
    # 이건 real world 환경 보면서 적당히 맞춰줘야될듯
    # 2. spline interpolation, kalman filter (noise reduction), other shits
    # 3. 저 1번이 있는 이유도 가끔 점이 너무 튀어나가서 그런건데, 사실 진짜 원인은 손 일부가 화면 밖으로 나갔을 때도 손을 트래킹하는데 이게 실제 손이랑 맞지가 않음
    # 화면 밖으로 너무 튀어나가면 트래킹을 멈추게 하자
    if type(hand_3d) != type(None):
        # 8번점 = 검지끝
        # TODO 이거 바꿔야됨
        # 점 하나의 좌표값 기반 -> 점 두개의 거리 기반(이동속도)

        if len(recent_points) == 0:
            recent_points.append(hand_3d[0].points[8])
            recent_timings.append(frame_no / FRAME_RATE)

        else:
            speed = np.linalg.norm(hand_3d[0].points[8] - recent_points[-1]) / ((frame_no / FRAME_RATE) - recent_timings[-1])
            if np.linalg.norm(hand_3d[0].points[8]) <= DIST_THRESHOLD and speed <= SPEED_THRESHOLD and np.abs(hand_3d[0].points[8][2]) >= DEPTH_MIN_THRESHOLD:
                cur_points = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array([hand_3d[0].points[8], recent_points[-1]])))
                # vis.add_geometry(cur_points)
                line = o3d.geometry.LineSet()
                line.points = cur_points.points
                line.lines = o3d.utility.Vector2iVector([[0, 1]])
                # vis.add_geometry(line)
                recent_points.append(hand_3d[0].points[8])
                recent_timings.append(frame_no / FRAME_RATE)

        while len(recent_points) > 5:
            recent_points.popleft()
            recent_timings.popleft()

        # spline interpolation을 위해서는 기본 4개의 점이 필요 + 1개는 안정성을 위해 추가
        if len(recent_points) == 5:
            t = np.array(recent_timings)
            dt = t[1:] - t[:-1]

            # 손가락이 화면에서 너무 오랬동안 없어져 있던 경우에는 보간 중지
            if max(dt) <= .5:
                xy = np.array(recent_points)
                tt = np.linspace(recent_timings[-2], recent_timings[-1], 30)
                bspl = sp.interpolate.make_interp_spline(t, xy)
                points = o3d.utility.Vector3dVector(bspl(tt))
                pointcloud = o3d.geometry.PointCloud(points)
                vis.add_geometry(pointcloud)
        
        # vis.add_geometry(hand_3d[0])
        # vis.add_geometry(hand_3d[1])
        vis.poll_events()
        vis.update_renderer()

        # vis.remove_geometry(hand_3d[0])
        # vis.remove_geometry(hand_3d[1])

    # Flip the image horizontally for a selfie-view display.
    image_full = np.concatenate((image_l, image_r), axis=1)
    cv2.imshow('MediaPipe Hands',image_full)

    key = cv2.waitKey(1)
    if key & 0xFF == 27:
        break
    elif key & 0xFF == ord('p'):
        cv2.waitKey(0)

    frame_no += 1

    print(f"FPS: {1 / (time.time() - time_elapsed):.3f}")
    print('-'*20)
cap_l.release()
cap_r.release()
cv2.destroyAllWindows()

vis.run()
