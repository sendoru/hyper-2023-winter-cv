import pickle
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

def preprocess(image: np.ndarray):
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
        print('-----------------------')
        print(hand_landmarks)
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
    
# skip first 2 seconds
for i in range(60):
    cap_l.read()
    cap_r.read()

success_l, image_l = cap_l.read()
success_r, image_r = cap_r.read()
img_h = image_l.shape[0]
img_w = image_l.shape[1]

results_l = preprocess(image_l)
results_r = preprocess(image_r)

cv2.imwrite("image_l.jpg", image_l)
cv2.imwrite("image_r.jpg", image_r)

log = open("log.txt", "w")
print("image_l", file=log)
hand_landmarks_l, hand_landmarks_r = None, None
for hand_no, hand_landmarks in enumerate(results_l.multi_hand_landmarks):
    print(f'HAND NUMBER: {hand_no+1}')
    print('-----------------------')
    print(hand_landmarks, file=log)
print()

print("image_r", file=log)
for hand_no, hand_landmarks in enumerate(results_r.multi_hand_landmarks):
    print(f'HAND NUMBER: {hand_no+1}')
    print('-----------------------')
    print(hand_landmarks, file=log)

log.close()

cap_l.release()
cap_r.release()