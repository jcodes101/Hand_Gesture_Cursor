## Libraries:

- 1. OpenCV:
     for computer vision, capturing live video from webcam and processing frames

- 2. MediaPipe:
     used for real-time hand tracking, detecting landmarks of fingers

- 3. PyAutoGUI:
     for controlling the mouse/cursor and performing clicks based on hand gestures

- 4. Pynput:
     for advanced mouse/keyboard input events (optional but is used for clicks and scrolling)

- 5. NumPy:
     for numerical operations, distance calculations, and array handling

- 6. Time:

  # used to track time during execution

- 7. Math:
     used to calculate distances and angles between hand landmarks

## How it works

1. open the webcam
   - access the webcam utilizing OpenCV

2. use landmarks with MediaPipe
   - detect hand landmarks (key points)

3. track hand movements
   - landmarks help to track a hands position in real time

4. connect hand to cursor
   - map the hands movemnet with the cursor

5. control actions with gestures
   - clicks and scrolls are controlled using the hand
