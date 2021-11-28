##########################################################################################
# Standard Library imports
import cv2 as cv
import mediapipe as mp
import numpy as np
from enum import Enum

##########################################################################################
# Threshold value for distance between relevant landmarks to check if finger is up
DISTANCE_THRESHOLD = 40

##########################################################################################
# enum definitions
class Hands(Enum):
    Left = 0
    Right = 1

class Coordinates(Enum):
    X = 0
    Y = 1

class Fingers(Enum):
    Thumb = 4
    Index = 8
    Middle = 12
    Ring = 16
    Little = 20

##########################################################################################
class HandTracker:
    # class constructor
    def __init__(self, mode=False, max_hands=2, detect_confidence=0.5, track_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detect_confidence = detect_confidence
        self.track_confidence = track_confidence
        self.results = 0

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.max_hands,
                                        min_detection_confidence=self.detect_confidence,
                                        min_tracking_confidence=self.track_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    # detect hands in the given frame and draw landmarks if requested
    def find_hands(self, image, draw_landmarks=True):
        img_RGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(img_RGB)

        if draw_landmarks:
            if self.results.multi_hand_landmarks:
                for handlandmarks in self.results.multi_hand_landmarks:
                    self.mpDraw.draw_landmarks(image, handlandmarks, self.mpHands.HAND_CONNECTIONS)

        return image

    # from the given frame and co-ordinates, segregate landmark information based on the hand it belongs to
    def find_position(self, image):
        lm_list = [{}, {}]
        if self.results.multi_hand_landmarks:
            for hand_index, each_hand_landmarks in enumerate(self.results.multi_hand_landmarks):
                for tip_id, lm in enumerate(each_hand_landmarks.landmark):
                    h, w, c = image.shape
                    x, y = int(lm.x * w), int(lm.y * h)
                    # mitigate error conditions from mediapipe which sometimes generates wrong hands information
                    if hand_index < self.max_hands:
                        lm_list[hand_index][tip_id] = (x, y)

        return lm_list

    # A finger is up if the co-ordinate of the fingertip is above the landmark point just below the tip
    # Hand specific information is determined based on the position of the wrist and thumb landmarks
    # if the X-coordinate of the wrist is less than that of the thumb, its right hand. Else left hand
    def detect_fingers_up(self, image, lm_list):
        total_finger_count = 0
        fingers_info = [[], []]
        finger_tips = [finger.value for finger in Fingers]

        for each_hand_dict in lm_list:
            if len(each_hand_dict):
                for each_lm_id in each_hand_dict:
                    if each_lm_id in finger_tips:
                        hand_info = Hands.Left.value
                        if each_hand_dict[0][Coordinates.X.value] < each_hand_dict[1][Coordinates.X.value]:
                            hand_info = Hands.Right.value

                        point_primary = each_hand_dict[each_lm_id]
                        point_secondary = each_hand_dict[each_lm_id-2]

                        if self.confidence_calculator(point_primary, point_secondary, each_lm_id, hand_info):
                            total_finger_count += 1
                            cv.circle(image, point_primary, 7, (255, 0, 0), cv.FILLED)
                            cv.circle(image, point_secondary, 7, (0, 255, 0), cv.FILLED)
                            cv.line(image, point_primary, point_secondary, (255, 255, 255), 2)
                            fingers_info[hand_info].append(each_lm_id)

        return total_finger_count, fingers_info

    # check if the confidence is high about the finger being up
    # coordinate value to measure confidence:
    # X for thumb since its movement is horizontal
    # Y for others since they move vertically
    def confidence_calculator(self, point_1, point_2, finger, hand_info):
        finger_up = False

        if finger == Fingers.Thumb.value:
            distance = self.calculate_distance_between_points(point_1, point_2)

            if hand_info == Hands.Left.value:
                if point_1[Coordinates.X.value] < point_2[Coordinates.X.value] and distance > DISTANCE_THRESHOLD:
                    finger_up = True
            else:
                if point_1[Coordinates.X.value] > point_2[Coordinates.X.value] and distance > DISTANCE_THRESHOLD:
                    finger_up = True
        else:
            distance = self.calculate_distance_between_points(point_1, point_2)
            if distance > DISTANCE_THRESHOLD-10 and point_1[Coordinates.Y.value] < point_2[Coordinates.Y.value]:
                finger_up = True

        return finger_up

    # calculate the euclidean distance between any 2 given cartesian coordinates
    @staticmethod
    def calculate_distance_between_points(point_1, point_2):
        x_1 = point_1[0]
        y_1 = point_1[1]
        x_2 = point_2[0]
        y_2 = point_2[1]

        distance = np.sqrt(((x_2 - x_1)**2) + ((y_2 - y_1)**2))

        return distance

    # calculate the euclidean distance and interpolate it to the range 0 - 100
    def distance_evaluator(self, image, co_ords_list, hand_range=(0, 0)):
        thumb_coords = (0, 0)
        index_coords = (0, 0)
        distance = -1

        if not(self.left_hand_in_frame(co_ords_list) or self.no_hand_in_frame(co_ords_list)):
            index_coords = co_ords_list[0][Fingers.Index.value]
            thumb_coords = co_ords_list[0][Fingers.Thumb.value]
            cv.circle(image, index_coords, 7, (255, 0, 0), cv.FILLED)
            cv.circle(image, thumb_coords, 7, (255, 0, 0), cv.FILLED)
            cv.line(image, thumb_coords, index_coords, (255, 255, 255), 2)

            distance = self.calculate_distance_between_points(thumb_coords, index_coords)
            distance = np.interp(distance, hand_range, (0, 100))

        return int(distance)

    # check if left hand is in frame
    @staticmethod
    def left_hand_in_frame(coordinates):
        for each_hand_info in coordinates:
            if each_hand_info:
                if each_hand_info[0][Coordinates.X.value] > each_hand_info[1][Coordinates.X.value]:
                    return True

        return False

    # check if no hands are in the frame
    @staticmethod
    def no_hand_in_frame(coordinates):
        if not coordinates[0] and not coordinates[1]:
            return True

        return False
