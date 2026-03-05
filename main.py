import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui

# l oads the trained ML model file (TensorFlow Lite) and contains network to detect hands
# pass in and configure model to detect up to 2 hands
hand_model = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=hand_model,
    num_hands=2
)
detector = vision.HandLandmarker.create_from_options(options)

# mapping hand coordinates for screen control
    # ex. Example: If your camera sees your hand at the far right, 
    # you want your mouse to go to the far right of the screen.
screen_w, screen_h = pyautogui.size()
print(f"\n hand mouse control .")

# store previos cursor position to allow smooth movement
prev_screen_x, prev_screen_y = 0, 0

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("can't open camera")
    exit()

while True:
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
            for landmark in hand_landmarks:
                # convert coordinates to pixel coordinates
                px = int(landmark.x * w)
                py = int(landmark.y * h)

                # draws dot at each landmaerk
                # image, (x,y), radius, color, thickness
                cv2.circle(camera_frame, (px, py), 5, (0, 255, 0), -1)
                print(landmark.x, landmark.y)

                # =========set finger tip landmarks=================
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]
                middle_tip = hand_landmarks.landmark[12]
                ring_tip = hand_landmarks.landmark[16]
                pinky_tip = hand_landmarks.landmark[20]
                # ==================================================

                # list comprehension that outputs 1s (finger up) and 0s (finger down)
                #   tip -> is the fingertip landmark
                #   tip - 2 -> the landmark two joints below the tip (near knuckle)
                #       hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y 
                #           -> if the tip is above the knuckle, the finger is considered "up"
                gesture_determination = [
                    1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y else 0
                    for tip in [8,12,16,20]
                ]


    cv2.imshow("live video", camera_frame)
    if(cv2.waitKey(1)==ord('q')):
        break
camera.release()
cv2.destroyAllWindows()

