import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import time
import math


hand_model = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=hand_model,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

# camera to screen — for mapping hand coordinates for screen control
screen_w, screen_h = pyautogui.size()
print(f"\n hand mouse control .")

# store previos cursor position to allow smooth movement
prev_screen_x, prev_screen_y = pyautogui.position()

# gesture time control
click_start_time = None
click_times = []
click_cooldown = 0.5
scroll_mode = False
frozen_cursor = False
scroll_time = 0
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

# message display time
status_text = ""
status_time = 0

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not camera.isOpened():
    print("can't open camera")
    exit()

while True:
    now = time.time()
    success, frame = camera.read()
    if not success:
        print("can't recieve frame")
        break
    camera_frame = cv2.flip(frame, 1)

    # convert color format | cv: BGR - mp: RGB
    # prepare image for the mp model and run the model on that specific image and pass it in to result
    rgb_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image (
        image_format = mp.ImageFormat.SRGB,
        data = rgb_frame
    )
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        # get image dimensions
        h, w, _ = camera_frame.shape

        for hand_landmarks in result.hand_landmarks:
            # only draw important hand landmarks (finger tips) to minimize the amount of circles drawn every frame
            important_landmarks = [4, 8, 12, 16, 20]
            
            # skeleton connections (mediapipe hand structure)
            connections = [
                (0,1),(1,2),(2,3),(3,4),                        # thumb
                (0,5),(5,6),(6,7),(7,8),                        # index
                (5,9),(9,10),(10,11),(11,12),                   # middle
                (9,13),(13,14),(14,15),(15,16),                 # ring
                (13,17),(17,18),(18,19),(19,20),                # pinky
                (0,17)                                          # palm base
            ]

            for start, end in connections:
                x1 = int(hand_landmarks[start].x * w)
                y1 = int(hand_landmarks[start].y * h)
                x2 = int(hand_landmarks[end].x * w)
                y2 = int(hand_landmarks[end].y * h)

                # skeleton line
                cv2.line(camera_frame, (x1,y1), (x2,y2), (255, 255, 255), 1)
            
            for l in important_landmarks:
                landmark = hand_landmarks[l]
                # convert coordinates to pixel coordinates
                px = int(landmark.x * w)
                py = int(landmark.y * h)

                # draws dot at each landmaerk
                # image, (x,y), radius, color, thickness
                # cv2.circle(camera_frame, (px, py), 8, (120, 0, 120), -1)
                cv2.circle(camera_frame, (px, py), 6, (245, 162, 130), -1)
                # print(landmark.x, landmark.y)

            # =========set finger tip landmarks=================
            thumb_tip = hand_landmarks[4]
            index_tip = hand_landmarks[8]
            middle_tip = hand_landmarks[12]
            ring_tip = hand_landmarks[16]
            pinky_tip = hand_landmarks[20]
            # ==================================================

                # list comprehension that outputs 1s (finger up) and 0s (finger down)
                #   tip -> is the fingertip landmark
                #   tip - 2 -> the landmark two joints below the tip (near knuckle)
                #       hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y 
                #           -> if the tip is above the knuckle, the finger is considered "up"
            gesture_determination = [
                1 if hand_landmarks[tip].y < hand_landmarks[tip - 2].y else 0
                for tip in [8,12,16,20]
            ]

            # distance between thumb and index
            distance = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)

            if distance < 0.06:
                if not frozen_cursor:
                    frozen_cursor = True
                    click_times.append(now)

                    if len(click_times) > 2:
                        click_times.pop(0)

                    # double click check
                    if len(click_times) >= 2 and click_times[-1] - click_times[-2] < 0.7:
                        pyautogui.doubleClick()
                        status_text = "double click"
                        status_time = now
                        click_times = []
                    else:
                        pyautogui.click()
                        status_text = "single click"
                        status_time = now
            else:
                frozen_cursor = False  

            # move cursor with index_finger
            deadzone = 5

            if not frozen_cursor:
                # modified mapping | calculate screen pixel location
                #   ex. screen_x = 0.5 * 1920 = 960
                target_x = int((index_tip.x - 0.1) * screen_w * 1.25)
                target_y = int((index_tip.y - 0.1) * screen_h * 1.25)

                dynamic_x = abs(target_x - prev_screen_x)
                dynamic_y = abs(target_y - prev_screen_y)
                speed = dynamic_x + dynamic_y

                alpha = min(0.35, max(0.08, speed / 1000))

                # exponential smoothing (ES)
                #   previous value contributes exponentially less over time
                screen_x = int(alpha * target_x + (1 - alpha) * prev_screen_x)
                screen_y = int(alpha * target_y + (1 - alpha) * prev_screen_y)

                # velocity prediction, removing micro lag
                velocity_x = screen_x - prev_screen_x
                velocity_y = screen_y - prev_screen_y

                screen_x += velocity_x * 0.3
                screen_y += velocity_y * 0.3

                # deadzone filter
                if abs(screen_x - prev_screen_x) > deadzone or abs(screen_y - prev_screen_y) > deadzone:
                    pyautogui.moveTo(screen_x, screen_y)
                    prev_screen_x, prev_screen_y = screen_x, screen_y

            # scrolling mode
            if sum(gesture_determination) == 4:
                scroll_mode = True
            else:
                scroll_mode = False

            # scroll actions
            if scroll_mode and now - scroll_time > 0.2:
                if index_tip.y < 0.4:
                    pyautogui.scroll(60)
                    status_text = "scrolling up"
                    status_time = now
                    scroll_time = now
                elif index_tip.y > 0.6:
                    pyautogui.scroll(-60)
                    status_text = "scrolling down"
                    status_time = now
                    scroll_time = now

    # display single|double|scrolling_up|scrolling_down click based on hand genstures
    if now - status_time < 1:
            cv2.putText(camera_frame, status_text, (10,50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    cv2.imshow("live video", camera_frame)
    if(cv2.waitKey(1)==ord('q')):
        break
camera.release()
cv2.destroyAllWindows()

