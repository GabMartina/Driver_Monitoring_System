import cv2

import mediapipe as mp

import numpy as np

import time

import statistics as st

import os

from math import sqrt

PIX_NEEDED_FOR_EYE_DETECTION = 20
CLOSED_EYES_THRESH_PERCLOS = 0.7
ALERT_THRESH = 0.8
TIME_WINDOW = 10 

HEAD_ROLL_THRESH = 30 
HEAD_PITCH_THRESH = 50
HEAD_YAW_THRESH = 40 

EYES_PITCH_THRESH = 90
EYES_YAW_THRESH = 50

# Wrapper for head and gaze angles
class AlertFlags:
    def __init__(self):
        self.drowsiness = 0 # if 0 either not enough pixer or no need to alert
        self.head = {
            "roll": 0,
            "pitch": 0,
            "yaw": 0
        }
        self.eyes = {
            "pitch": {
                "left": 0,
                "right": 0,
                "avg": 0
            },
            "yaw": {
                "left": 0,
                "right": 0,
                "avg": 0
            }
        }

    def reset(self):
        self.drowsiness = 0
        self.head["roll"] = 0
        self.head["pitch"] = 0
        self.head["yaw"] = 0
        self.eyes["pitch"]["left"] = 0
        self.eyes["pitch"]["right"] = 0
        self.eyes["pitch"]["avg"] = 0
        self.eyes["yaw"]["left"] = 0
        self.eyes["yaw"]["right"] = 0
        self.eyes["yaw"]["avg"] = 0
    
# Wrapper for Perclos evaluation
class Perclos:
    def __init__(self):
        self.win = []
        self.start = -1 
        self.sat_min = 0
        self.sat_max = 1
        self.min = 1
        self.max = 0

    def set_start(self, start):
        self.start = start

    def ready(self):
        return True if (time.time()-self.start) > TIME_WINDOW else False

    def push_EAR(self, EAR, timestamp):
        # protection for noise out of scale
        if EAR < self.sat_min: EAR = self.sat_min
        if EAR > self.sat_max: EAR = self.sat_max
        self.win.append((EAR, timestamp))

        # update min/max
        if EAR > self.max and EAR < self.sat_max: self.max = EAR
        if EAR < self.min and EAR > self.sat_min: self.min = EAR

    def clean_window(self):
        now = time.time()
        self.win = [el for el in self.win if clean_time_window(el, now)]
    
    def get_values(self):
        return [v for (v,_) in self.win]
    
    def get_normalised_values(self):
        values = self.get_values()
        z = self.max-self.min
        z = z if z > 0 else self.sat_max-self.sat_min
        return [v/z for v in values]
    
    def need_to_alert(self):        
        self.clean_window()
        normalised_values = self.get_normalised_values()
        count_under_thresh = [1 for x in normalised_values if x < CLOSED_EYES_THRESH_PERCLOS]
        return True if sum(count_under_thresh)/len(normalised_values) > ALERT_THRESH else False


def norm2(p1, p2) :
    x1, y1 = p1
    x2, y2 = p2
    return sqrt((y2-y1)**2 + (x2-x1)**2)


def clean_time_window(item, now):
    [_, timestamp] = item
    return True if timestamp >= (now - TIME_WINDOW) else False


def calculate_EAR(p):
    # p = [p1, p2, p3, p4, p5, p6]
    x1, _ = p[0]
    x4, _ = p[3]
    _, y2 = p[1]
    _, y3 = p[2]
    _, y5 = p[4]
    _, y6 = p[5]
    return (abs(y2 - y6) + abs(y3 - y5)) / (2*abs(x1 - x4))


def  head_angles(face_2d, face_3d, img_h, img_w):
    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    # Camera matrix
    focal_length = 1*img_w
    cam_matrix = np.array([[focal_length, 0, img_h/2],
                            [0, focal_length, img_w/2],
                            [0, 0, 1]])

    # Distorsion parameters
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    # Solve PnP
    _, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

    # Get rotational matrix
    rmat, _ = cv2.Rodrigues(rot_vec)

    # Get angles
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    # Convert the angles to degrees
    pitch = angles[0] * 900 # 900 to have a better approximation
    yaw = -angles[1] * 900
    roll = 180+(np.arctan2(point_RER[1]-point_LEL[1], point_RER[0]-point_LEL[0])*180 / np.pi)
    # Scale between 0-180
    if roll > 180: roll -= 360 
    
    return roll, pitch, yaw


def offset(p1, p2):
        horizontal = p2[0] - p1[0]
        vertical = p1[1] - p2[1]
        return (horizontal, vertical)


def alpha(coord, type):
    if type == 'p': x, y = coord
    elif type == 'y': y, x = coord
    else: return

    alpha = np.arctan2(y, x) * 180 / np.pi
    return alpha


def pitch_compute(l, r):
    type = 'p'
    return (alpha(l, type), alpha(r, type))   


def yaw_compute(l, r):
    type = 'y'
    return (alpha(l, type), alpha(r, type))   


def avg(p):
    return (p[0]+p[1])/2


# Check head is within thresholds
def head_out_of_bounds(angles):
    if abs(angles.head["pitch"]) > HEAD_PITCH_THRESH:
        return True
    if abs(angles.head["roll"]) > HEAD_ROLL_THRESH:
        return True
    if abs(angles.head["yaw"]) > HEAD_YAW_THRESH:
        return True
    return False        


# Check gaze is within threasholds (separately for x-axis and y-axis)
# Return true if no alert needed
def gaze_to_center(angles):
    x, y = True, True
    if abs(angles.eyes["yaw"]["avg"] > EYES_YAW_THRESH):
        x = False
    if abs(angles.eyes["pitch"]["avg"] > EYES_PITCH_THRESH):
        y = False
    return x, y 


# With yawed head, check gaze along x-axis
# Return true if no alert needed
def head_xaxis(angles):
    x = True

    # head left eyes right
    if angles.head["yaw"] < 0:
            if alerts.eyes["yaw"]["right"] < 0:
                x = False
    # head right eyes left
    if alerts.head["yaw"] > 0:
            if alerts.eyes["yaw"]["left"] > 0:
                x = False
    
    return x


# With pitched head, check gaze along y-axis
# Return true if no alert needed
def head_yaxis(angles):
    y = True

    # head up eyes down
    if angles.head["pitch"] < 0:
            if alerts.eyes["pitch"]["avg"] < 0:
                y = False
    # heads down eyes up
    if alerts.head["pitch"] > 0:
            if alerts.eyes["pitch"]["avg"] > 0:
                y = False
    
    return y


# Evaluate need to alert in case of driver distraction
# Return true if alert needed
def evaluate_distraction(angles):
    if head_out_of_bounds(angles):
        return True
    else:
        ok_xaxis, ok_yaxis = False, False

        ok_xaxis, ok_yaxis = gaze_to_center(angles)

        ok_xaxis = head_xaxis(angles)

        ok_yaxis = head_yaxis(angles)
        
        # If one between ok_xaxis and ok_yaxis is still false
        # the driver is looking left/right (x-axis) or up/down (y-axis) 
        # in an unacceptable way
        if ok_xaxis and ok_yaxis:
            return False
        
    return True


# 2 - Set the desired setting

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(

    max_num_faces=1,

    refine_landmarks=True,  # Enables  detailed eyes points

    min_detection_confidence=0.5,

    min_tracking_confidence=0.5

)

mp_drawing_styles = mp.solutions.drawing_styles

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Get the list of available capture devices (comment out)

# index = 0

# arr = []

# while True:

#    dev = cv2.VideoCapture(index)

#    try:

#        arr.append(dev.getBackendName)

#    except:

#        break

#    dev.release()

#    index += 1

# print(arr)


# 3 - Open the video source

cap = cv2.VideoCapture(0)  # Local webcam (index start from 0)

# 4 - Iterate (within an infinite loop)

perclos = Perclos() # Create data structure for PERCLOS
alerts = AlertFlags() # Create data structure for evaluating need to alert

count = -1

while cap.isOpened():
    count += 1

    # 4.1 - Get the new frame

    success, image = cap.read()

    start = time.time()


    # Also convert the color space from BGR to RGB

    if image is None:
        break

        # continue

    # else: #needed with some cameras/video input format

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # To improve performace

    image.flags.writeable = False

    # 4.2 - Run MediaPipe on the frame

    results = face_mesh.process(image)

    # To improve performance

    image.flags.writeable = True

    # Convert the color space from RGB to BGR

    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    face_2d = []        # 2D points of the face (x, y)
    face_3d = []        # 3D points of the face (x, y, z)

    img_h, img_w, img_c = image.shape

    point_RER = []  # Right Eye Right

    point_REB = []  # Right Eye Bottom

    point_REL = []  # Right Eye Left

    point_RET = []  # Right Eye Top

    point_LER = []  # Left Eye Right

    point_LEB = []  # Left Eye Bottom

    point_LEL = []  # Left Eye Left

    point_LET = []  # Left Eye Top

    point_REIC = []  # Right Eye Iris Center

    point_LEIC = []  # Left Eye Iris Center

    # 4.3 - Get the landmark coordinates

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            for idx, lm in enumerate(face_landmarks.landmark):

                # Eye Gaze (Iris Tracking)

                # Left eye indices list

                # LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398 ]

                # Right eye indices list

                # RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]

                # LEFT_IRIS = [473, 474, 475, 476, 477]

                # RIGHT_IRIS = [468, 469, 470, 471, 472]

                if idx == 33:
                    point_RER = (lm.x * img_w, lm.y * img_h)
                    # p4
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)

                if idx == 145:
                    point_REB = (lm.x * img_w, lm.y * img_h)

                    # cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)

                if idx == 133:
                    point_REL = (lm.x * img_w, lm.y * img_h)
                    # p1
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)

                if idx == 159:
                    point_RET = (lm.x * img_w, lm.y * img_h)

                    # cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)

                if idx == 158:
                    point_RETL = (lm.x * img_w, lm.y * img_h)
                    # p2
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)
                
                if idx == 160:
                    point_RETR = (lm.x * img_w, lm.y * img_h)
                    # p3
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)

                if idx == 153:
                    point_REBL = (lm.x * img_w, lm.y * img_h)
                    # p6
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)

                if idx == 144:
                    point_REBR = (lm.x * img_w, lm.y * img_h)
                    # p5
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)

                if idx == 362:
                    point_LER = (lm.x * img_w, lm.y * img_h)
                    # this is left inner point (p1): 
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(255, 0, 0), thickness=-1)

                if idx == 374:
                    point_LEB = (lm.x * img_w, lm.y * img_h)
                    # this is middle bottom:
                    # cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(255, 0, 0), thickness=-1)

                if idx == 263:
                    point_LEL = (lm.x * img_w, lm.y * img_h)
                    # this is left outer (p4):
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(255, 0, 0), thickness=-1)

                if idx == 386:
                    point_LET = (lm.x * img_w, lm.y * img_h)
                    # this is middle top:
                    # cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(255, 0, 0), thickness=-1)

                if idx == 380:
                    point_LEBR = (lm.x * img_w, lm.y * img_h)
                    # this is p6:
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(255, 0, 0), thickness=-1)

                if idx == 373:
                    point_LEBL = (lm.x * img_w, lm.y * img_h)
                    # this is p5:
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(255, 0, 0), thickness=-1)

                if idx == 385:
                    point_LETR = (lm.x * img_w, lm.y * img_h)
                    # this is p2 candidate:
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(255, 0, 0), thickness=-1)
                
                if idx == 387:
                    point_LETL = (lm.x * img_w, lm.y * img_h)
                    # this is p3 candidate:
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(255, 0, 0), thickness=-1)
                
                if idx == 468:
                    point_REIC = (lm.x * img_w, lm.y * img_h)
                    # center of right eye
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(255, 255, 0), thickness=-1)

                if idx == 469:
                    point_469 = (lm.x * img_w, lm.y * img_h)

                    # cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 255, 0), thickness=-1)

                if idx == 470:
                    point_470 = (lm.x * img_w, lm.y * img_h)

                    # cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 255, 0), thickness=-1)

                if idx == 471:
                    point_471 = (lm.x * img_w, lm.y * img_h)

                    # cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 255, 0), thickness=-1)

                if idx == 472:
                    point_472 = (lm.x * img_w, lm.y * img_h)

                    # cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 255, 0), thickness=-1)

                if idx == 473:
                    point_LEIC = (lm.x * img_w, lm.y * img_h)
                    # center of left eye
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 255, 255), thickness=-1)

                if idx == 474:
                    point_474 = (lm.x * img_w, lm.y * img_h)

                    # cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(255, 0, 0), thickness=-1)

                if idx == 475:
                    point_475 = (lm.x * img_w, lm.y * img_h)

                    # cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(255, 0, 0), thickness=-1)

                if idx == 476:
                    point_476 = (lm.x * img_w, lm.y * img_h)

                    # cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(255, 0, 0), thickness=-1)

                if idx == 477:
                    point_477 = (lm.x * img_w, lm.y * img_h)

                    # cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(255, 0, 0), thickness=-1)

                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:

                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)

                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x,y,lm.z])

                # LEFT_IRIS = [473, 474, 475, 476, 477]

                if idx == 473 or idx == 362 or idx == 374 or idx == 263 or idx == 386:  # iris points

                    # if idx == 473 or idx == 474 or idx == 475 or idx == 476 or idx == 477: # eye border

                    if idx == 473:
                        left_pupil_2d = (lm.x * img_w, lm.y * img_h)

                        left_pupil_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)


                # RIGHT_IRIS = [468, 469, 470, 471, 472]

                if idx == 468 or idx == 33 or idx == 145 or idx == 133 or idx == 159:  # iris points

                    # if idx == 468 or idx == 469 or idx == 470 or idx == 471 or idx == 472: # eye border

                    if idx == 468:
                        right_pupil_2d = (lm.x * img_w, lm.y * img_h)

                        right_pupil_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)


            # 4.4. - Draw the positions on the frame

            l_eye_width = point_LEL[0] - point_LER[0]

            l_eye_height = point_LEB[1] - point_LET[1]

            l_eye_center = [(point_LEL[0] + point_LER[0]) / 2, (point_LEB[1] + point_LET[1]) / 2]

            # cv2.circle(image, (int(l_eye_center[0]), int(l_eye_center[1])), radius=int(horizontal_threshold * l_eye_width), color=(255, 0, 0), thickness=-1) #center of eye and its radius

            cv2.circle(image, (int(point_LEIC[0]), int(point_LEIC[1])), radius=2, color=(0, 255, 0),
                       thickness=-1)  # Center of iris

            cv2.circle(image, (int(l_eye_center[0]), int(l_eye_center[1])), radius=2, color=(128, 128, 128),
                       thickness=-1)  # Center of eye

            # print("Left eye: x = " + str(np.round(point_LEIC[0],0)) + " , y = " + str(np.round(point_LEIC[1],0)))

            r_eye_width = point_REL[0] - point_RER[0]

            r_eye_height = point_REB[1] - point_RET[1]

            r_eye_center = [(point_REL[0] + point_RER[0]) / 2, (point_REB[1] + point_RET[1]) / 2]

            # cv2.circle(image, (int(r_eye_center[0]), int(r_eye_center[1])), radius=int(horizontal_threshold * r_eye_width), color=(255, 0, 0), thickness=-1) #center of eye and its radius

            cv2.circle(image, (int(point_REIC[0]), int(point_REIC[1])), radius=2, color=(0, 0, 255),
                       thickness=-1)  # Center of iris

            cv2.circle(image, (int(r_eye_center[0]), int(r_eye_center[1])), radius=2, color=(128, 128, 128),
                       thickness=-1)  # Center of eye

            # speed reduction (comment out for full speed)
            
            # time.sleep(1 / 25)  # [s]
                    

        end = time.time()

        totalTime = end - start

        if totalTime > 0:

            fps = 1 / totalTime

        else:

            fps = 0

        # print("FPS:", fps)
        # cv2.putText(image, f'FPS : {int(fps)}', (350, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        alerts.reset()

        # DROWSINESS -->

        # Check that eye has enough pixels to be evaluated
        if norm2(point_REL, point_RER) < PIX_NEEDED_FOR_EYE_DETECTION:
            cv2.putText(image, "Get closer to the camera", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # Calculate Eye Aspect Ratio (EAR)
            EAR_RE = calculate_EAR([point_REL, point_RETL, point_RETR, point_RER, point_REBR, point_REBL]) # right eye
            EAR_LE = calculate_EAR([point_LER, point_LETR, point_LETL, point_LEL, point_LEBL, point_LEBR]) # left eye

            # Start taking time for time-window if not done yet
            if (perclos.start == -1): 
                perclos.set_start(time.time()) 

            # Store lowest EAR between left/right eye
            perclos.push_EAR(min(EAR_LE, EAR_RE), time.time()) 

            # Only when we have at least TIME_WINDOW seconds to evaluate
            # evaluate the need to alert for drowsy driver
            if perclos.ready() and perclos.need_to_alert():
                alerts.drowsiness = 1
                
            # debug
            # val = perclos.get_normalised_values()
            # format_v = [f'{v:.2f}' for v in val]
            # cv2.putText(image, " ".join(format_v), (100, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
            # cv2.putText(image, "EAR left: "+f'{EAR_LE:.2f}'+" right: "+f'{EAR_RE:.2f}' , (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)    

        # HEAD ROLL-PITCH-YAW -->
        roll, pitch, yaw = head_angles(face_2d, face_3d, img_h, img_w)
        alerts.head["roll"] = roll
        alerts.head["pitch"] = pitch
        alerts.head["yaw"] = yaw

        # debug
        # cv2.putText(image, f'Roll: {roll:.2f}, Pitch: {pitch:.2f}, Yaw: {yaw:.2f}', (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # EYES PITCH-YAW -->

        # Calculate (dx, dy) offset between center of the iris and eye
        offset_LE = offset(point_LEIC, l_eye_center) # left eye
        offset_RE = offset(point_REIC, r_eye_center) # right eye

        # debug
        # cv2.putText(image, f'HL: {offset_LE[0]:.2f}, HR: {offset_RE[0]:.2f}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        # cv2.putText(image, f'VL: {offset_LE[1]:.2f}, VR: {offset_RE[1]:.2f}', (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Calculate pitch/yaw separately for left/right eye
        eye_pitch = pitch_compute(offset_LE, offset_RE)
        eye_yaw = yaw_compute(offset_LE, offset_RE)

        # debug
        # cv2.putText(image, f'Left Eye Yaw: {eye_yaw[0]:.2f}, Pitch: {eye_pitch[0]:.2f}', (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # cv2.putText(image, f'Right Eye Yaw: {eye_yaw[1]:.2f}, Pitch: {eye_pitch[1]:.2f}', (20, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Calculate mean between left/right eye
        avg_eye_yaw = avg(eye_yaw)
        avg_eye_pitch = avg(eye_pitch)
        
        alerts.eyes["pitch"] = {"left": eye_pitch[0], "right": eye_pitch[1], "avg": avg_eye_pitch}
        alerts.eyes["yaw"] = {"left": eye_yaw[0], "right": eye_yaw[1], "avg": avg_eye_yaw}
        
        # debug
        # cv2.putText(image, f'Yaw AVG: {avg_eye_yaw:.2f}, Pitch AVG: {avg_eye_pitch:.2f}', (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if evaluate_distraction(alerts):
            cv2.putText(image, f'Warning: Driver is not paying attention!', (100, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)
        
        if alerts.drowsiness:
            cv2.putText(image, f'Warning: Drowsy driver!', (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)
                
        # 4.5 - Show the frame to the user

        cv2.imshow('Technologies for Autonomous Vehicles - Driver Monitoring Systems using AI code sample', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

# 5 - Close properly soruce and eventual log file

cap.release()

# log_file.close()


# [EOF]