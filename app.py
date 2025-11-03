#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointSequenceClassifier


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def main():
    # Argument parsing #################################################################
    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    # Use Holistic (gives hands + pose + face) but we only use hand landmarks here
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=use_static_image_mode,
        model_complexity=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # Only the sequence classifier is used for this workflow
    keypoint_sequence_classifier = KeyPointSequenceClassifier()

    # (no label CSVs used here — we'll display sequence-class ids directly)

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate / keypoint history
    history_length = 30
    # Store per-frame pre-processed landmark vectors (shape D)
    keypoint_history = deque(maxlen=history_length)

    #  ########################################################################
    mode = 0

    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True

        # Holistic provides left/right hand landmarks as separate attributes
        hand_landmarks = None
        if results.right_hand_landmarks:
            hand_landmarks = results.right_hand_landmarks
        elif results.left_hand_landmarks:
            hand_landmarks = results.left_hand_landmarks

        seq_pred = None

        # Bounding box and landmarks (only if hand detected)
        if hand_landmarks is not None:
            brect = calc_bounding_rect(debug_image, hand_landmarks)
            landmark_list = calc_landmark_list(debug_image, hand_landmarks)
        else:
            brect = [0, 0, 0, 0]
            landmark_list = []

        # Build per-frame feature vector (hands + upper-pose) and append to sequence buffer
        frame_features = extract_frame_landmarks(results)
        keypoint_history.append(frame_features)

        # If buffer is full, call the sequence classifier
        if len(keypoint_history) == history_length:
            try:
                seq_pred = keypoint_sequence_classifier(list(keypoint_history))
            except Exception as e:
                print(f"Sequence classifier error: {e}")

        # Drawing part
        debug_image = draw_bounding_rect(use_brect, debug_image, brect)
        debug_image = draw_landmarks(debug_image, landmark_list)

        # Display sequence prediction (if available)
        if seq_pred is not None:
            cv.putText(debug_image, f"SEQ:{seq_pred}", (10, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
            cv.putText(debug_image, f"SEQ:{seq_pred}", (10, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                       cv.LINE_AA)

        debug_image = draw_info(debug_image, fps, mode, number)

        # Screen reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def extract_frame_landmarks(results):
    """
    Build a per-frame feature vector from Holistic results.
    Format to match training model (477 features):
      - left hand: 21 landmarks × 3 = 63
      - right hand: 21 landmarks × 3 = 63
      - pose: ALL 33 landmarks × 3 = 99
      - face: 84 landmarks × 3 = 252
    Total: 63 + 63 + 99 + 252 = 477
    """
    frame_landmarks = []

    # Left hand (21 * 3 = 63)
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            frame_landmarks.extend([lm.x, lm.y, lm.z])
    else:
        frame_landmarks.extend([0.0] * 63)

    # Right hand (21 * 3 = 63)
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            frame_landmarks.extend([lm.x, lm.y, lm.z])
    else:
        frame_landmarks.extend([0.0] * 63)

    # Pose - ALL 33 landmarks (33 * 3 = 99)
    if results.pose_landmarks:
        for i in range(33):  # CHANGED from range(11)
            lm = results.pose_landmarks.landmark[i]
            frame_landmarks.extend([lm.x, lm.y, lm.z])
    else:
        frame_landmarks.extend([0.0] * 99)  # CHANGED from 33

    # Face landmarks - first 84 landmarks (84 * 3 = 252)
    if results.face_landmarks:
        # MediaPipe face mesh has 468 landmarks, we use first 84
        for i in range(min(84, len(results.face_landmarks.landmark))):
            lm = results.face_landmarks.landmark[i]
            frame_landmarks.extend([lm.x, lm.y, lm.z])
    else:
        frame_landmarks.extend([0.0] * 252)

    # Verify we have exactly 477 features
    if len(frame_landmarks) != 477:
        print(f"WARNING: Expected 477 features, got {len(frame_landmarks)}")
    
    return frame_landmarks


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


def draw_landmarks(image, landmark_point):
    # Accept either a mediapipe landmarks object (has .landmark) or
    # a list of (x, y) tuples / lists in image coordinates.
    if not landmark_point:
        return image

    img_h, img_w = image.shape[0], image.shape[1]

    # Build a list of (x, y) pixel coordinates
    points = []
    if hasattr(landmark_point, 'landmark'):
        # mediapipe LandmarkList
        for lm in landmark_point.landmark:
            x = min(int(lm.x * img_w), img_w - 1)
            y = min(int(lm.y * img_h), img_h - 1)
            points.append((x, y))
    else:
        # assume list-like of [x, y]
        for p in landmark_point:
            try:
                points.append((int(p[0]), int(p[1])))
            except Exception:
                # skip malformed entries
                continue

    # Use MediaPipe's hand connections for drawing skeleton lines if available
    try:
        connections = mp.solutions.hands.HAND_CONNECTIONS
    except Exception:
        connections = []

    # Draw connections (black thick, then white thin for a bordered look)
    for c in connections:
        start_idx, end_idx = c
        if start_idx < len(points) and end_idx < len(points):
            cv.line(image, points[start_idx], points[end_idx], (0, 0, 0), 6)
            cv.line(image, points[start_idx], points[end_idx], (255, 255, 255), 2)

    # Draw keypoints; make tip points larger
    tip_indices = {4, 8, 12, 16, 20}
    for idx, (x, y) in enumerate(points):
        radius = 8 if idx in tip_indices else 5
        cv.circle(image, (x, y), radius, (255, 255, 255), -1)
        cv.circle(image, (x, y), radius, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()
