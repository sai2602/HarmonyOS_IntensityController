##########################################################################################
# Standard Library imports
from HandTracking import HandTracker
import time
import cv2 as cv
import paho.mqtt.client as paho
import json

##########################################################################################
# Standard variables initialization
PUBLISH_WAIT_TIME = 1
COLOR_SELECT_WAIT_TIME = 10

# R: 0-255, G: 0-255, B: 0-255
colors = {"Default": [127, 127, 127],
          "Red": [255, 0, 0],
          "Green": [0, 255, 0],
          "Blue": [0, 0, 255]
          }

COLOR_SELECTED = False
COLOR_INFO_RECEIVED = False
number_selected = 0
RE_ENTRY = False

# payload contents for MQTT
data = json.loads('{"brightness":0, "rgb_color":[127, 127, 127]}')

broker = "192.168.0.163"
port = 1883


##########################################################################################
# MQTT client configuration for publishing data

def on_publish(client, userdata, result):
    print("data published \n")
    pass


my_client = paho.Client("Intensity")
my_client.on_publish = on_publish
my_client.username_pw_set(username="homeassistant", password="va4baaqueiJoosheiyee9Se9ohfoh4Pi3Ao"
                                                             "2aicoh6eepo4ahwa6Shohy4ohh9aZ")
my_client.connect(broker, port)


##########################################################################################
# extrapolate the range 0:100::0:255

def extrapolate_range(distance_value):
    return int(float(distance_value / 100.0) * 255.0)

##########################################################################################
# print user choice of color and return the corresponding key value

def selected_color_info(number):
    key_color = ""
    if not number or number > 3:
        print("Default color is selected")
        key_color = "Default"
    elif number is 1:
        print("Red color is selected")
        key_color = "Red"
    elif number is 2:
        print("Green color is selected")
        key_color = "Green"
    else:
        print("Blue color is selected")
        key_color = "Blue"

    return key_color


##########################################################################################
if __name__ == "__main__":
    cap = cv.VideoCapture(0)
    my_tracker = HandTracker(detect_confidence=0.7, track_confidence=0.7)

    # Min: least possible distance between thumb and index. Max: maximum possible distance between the fingers
    hand_range = [10, 150]

    prev_time = 0
    curr_time = 0
    publish_prev_time = time.process_time()

    while True:

        # read continuous video feed from system camera
        success, img = cap.read()

        # segment, detect and get the co-ordinates of the landmarks of the hands from the frame
        img = my_tracker.find_hands(img, draw_landmarks=False)
        co_ords_list = my_tracker.find_position(img)

        # wait for 10 seconds to get get a number from the user. Default number when no/wrong input is 0
        # 1: Red, 2: Green, 3: Blue. If 10s lapse or number is out of range, choose default color
        if time.process_time() - publish_prev_time < COLOR_SELECT_WAIT_TIME and not COLOR_INFO_RECEIVED:
            number_selected, _ = my_tracker.detect_fingers_up(img, co_ords_list)
            if number_selected:
                COLOR_INFO_RECEIVED = True

        else:
            # ensure static RGB value by checking for first entry into else section
            if not RE_ENTRY:
                key_color_dict = selected_color_info(number_selected)
                data["rgb_color"] = colors[key_color_dict]
                RE_ENTRY = True

            # calculate euclidean distance of the thumb and index fingertips to estimate intensity
            distance = my_tracker.distance_evaluator(img, co_ords_list, hand_range=hand_range)

            # publish the intensity by extrapolating the euclidean distance and publishing it after every second
            if distance != -1 and time.process_time() - publish_prev_time > PUBLISH_WAIT_TIME:
                extrapolated_distance = extrapolate_range(distance)
                data["brightness"] = extrapolated_distance
                ret = my_client.publish("light.projector_light", json.dumps(data))
                publish_prev_time = time.process_time()

        # calculate time difference between frames to get the frame rate (FPS)
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # add the FPS information to the video stream and show it on a window
        cv.putText(img, f'FPS: {(int(fps))}', (10, 70), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
        cv.imshow("hand", img)

        # stop complete execution of keyboard press "q" detected
        if cv.waitKey(1) & 0xFF == ord('q'):
            cv.destroyWindow("hand")
            break
