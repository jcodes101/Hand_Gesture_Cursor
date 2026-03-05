import cv2

'''
-   initialize a camera and check if it is
able to be opened

-   a constant infinite loop runs to keep the
camera running, and checks if running was a sucess

-   the frame is flipped to mirror properly for viewing
and the 'q' button is used to (break) the loop, and lastly
all cv2 windows are destroyed and the camera is released
'''
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
    cv2.imshow("live video", camera_frame)
    if(cv2.waitKey(1)==ord('q')):
        break
camera.release()
cv2.destroyAllWindows()