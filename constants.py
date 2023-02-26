import cv2
import numpy as np

L_INTRINSIC = np.array([[677.90820312, 0.,           319.61657153],
                        [ 0.,          621.58636475, 245.49386904],
                        [ 0.,          0.,           1. ]])

R_INTRINSIC = np.array([[667.01141357, 0.,          326.2987009 ],
                        [ 0.,          612.3939209, 233.17077425],
                        [ 0.,          0.,          1. ]])

ROT = np.array([[ 0.81654111, 0.00917727, 0.57721433],
                [-0.01709073, 0.99981965, 0.00828057],
                [-0.57703424, -0.01662644, 0.8165507 ]])

TRANS = np.array([[-325.37913832],
                [ 2.44065162],
                [ 96.19062845]])

L_DISTORTION = np.array([[ 1.46612706e-01,
                          -2.84449285e+00,
                          -2.98295340e-03,
                          1.18855292e-03,
                          1.76640972e+01]])

R_DISTORTION = np.array([[ 9.00673129e-02,
                          -1.87093373e+00,
                          -4.43220206e-03,
                          9.29939558e-03,
                          1.34399361e+01]])

L_EXTRINSIC = np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)

R_EXTRINSIC = np.concatenate((ROT.T, -TRANS), axis=1)

# 영상의 frame rate
# 실시간으로 처리되는 경우라면, fps도 실시간으로 구해서 그걸 대신 이용해야 함
# 단위: Hz
FRAME_RATE = 30
# 단위: mm/s
# 계산된 손가락의 이동 속도가 이것보다 크면 tracking 중지
# 근데 이것도 순간 속도 따질때지
# ex) 3초 내내 초속 4m으로 움직이진 않을거잖아
SPEED_THRESHOLD = 4000
# 
DIST_THRESHOLD = 4000

DEPTH_MIN_THRESHOLD = 50
