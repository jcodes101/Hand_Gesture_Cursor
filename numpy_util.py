import numpy as np

# used to detect if fingers are bent/straight
def get_angle(a, b, c):
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])

    angle = np.abs(np.degree(radians))

    return angle

# detect spread, pinch, or swipe gestures
def get_distance(landmark_list):
    if len(landmark_list) < 2:
        return
    (x1, y1), (x2, y2) = landmark_list[0], landmark_list[1]
    l = np.hypot(x2-x1, y2-y1)
    return np.interp(l, [0, 1], [0, 1000])


'''
- this numpy_util.py is used to convert raw hand landmark coordinates into meaningful
geometric data and this is the foundation of gesture recognition logic
'''
