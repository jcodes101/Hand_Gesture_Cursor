# Main Program Notes (hand_gesture_cursor)

Notes on crucial comments and behavior in `main.py`: hand detection, camera-to-screen mapping (including modified mapping), cursor smoothing (including dynamic alpha and velocity prediction), deadzone, gestures, skeleton overlay, and related logic.

---

## 1. Hand model and detection

- The program loads a trained ML model (TensorFlow Lite) from `hand_landmarker.task`. This model contains the network used to detect hands.
- `HandLandmarkerOptions` is configured with `num_hands=1` so only one hand is tracked.
- The detector is created once and reused each frame: `vision.HandLandmarker.create_from_options(options)`.

**Frame pipeline:** Camera frame (BGR) is converted to RGB because MediaPipe expects SRGB. The frame is wrapped in `mp.Image` and passed to `detector.detect(mp_image)` to get hand landmarks.

---

## 2. Coordinate mapping (camera to screen)

- Hand landmarks are in normalized coordinates (0–1). The goal is to map “where the camera sees the hand” to “where the mouse should be on the screen.”
- Screen size is taken from `pyautogui.size()` (`screen_w`, `screen_h`). The comment in code: “camera to screen — for mapping hand coordinates for screen control.”

### 2.1 Modified mapping (offset and scale)

The program uses a **modified mapping** instead of a direct 1:1 mapping from normalized coords to pixels:

old:

```python
target_x = int(index_tip.x * screen_w)
target_y = int(index_tip.y * screen_h)
```

new:

```python
target_x = int((index_tip.x - 0.1) * screen_w * 1.25)
target_y = int((index_tip.y - 0.1) * screen_h * 1.25)
```

**How it works:**

1. **Offset (`- 0.1`):** Subtracts 0.1 from both x and y. This shifts the “origin” of the hand so that the center of the camera view does not map to the center of the screen. In practice, it compensates for how the hand usually appears in frame (e.g. hand often slightly off-center or with margins), so the usable hand area maps to a better range on screen.
2. **Scale (`* 1.25`):** After the offset, the result is multiplied by 1.25. Normalized coords are in 0–1; after offset they can be negative or exceed 1. Multiplying by 1.25 expands the effective range, so small hand movements produce larger cursor movements and you can reach screen edges without moving your hand to the literal edges of the camera view.

**Example (x only, 1920 px wide):**

- Raw: `index_tip.x = 0.5` → `0.5 * 1920 = 960` (center).
- Modified: `(0.5 - 0.1) * 1920 * 1.25 = 0.4 * 2400 = 960` (still center for that value).
- For `index_tip.x = 0.9`: `(0.9 - 0.1) * 1920 * 1.25 = 1920` (far right). So the range 0.1–0.9 in normalized space is stretched to 0–1920 in pixels; you get full screen width from 80% of the normalized range.

**Summary:** Offset adjusts where the hand region maps; scale makes that region cover the full screen. Together they make cursor control feel less cramped and more aligned with the visible hand area.

---

## 3. Stored previous position (for smoothing)

- `prev_screen_x` and `prev_screen_y` store the last cursor position used for movement.
- They are updated only when a new smoothed position is actually applied, so the next frame’s smoothing uses the last “committed” position, not raw hand position. This is what allows smooth movement over time.

---

## 4. Regular smooth (linear interpolation, LERP)

The commented-out code in `main.py` uses **linear interpolation** between the previous cursor position and the target:

```python
# screen_x = int(prev_screen_x + (screen_x - prev_screen_x) * smooth)
# screen_y = int(prev_screen_y + (screen_y - prev_screen_y) * smooth)
```

- **Idea:** Move a fraction of the way from current position toward the target each frame. `smooth` (e.g. 0.2) is that fraction.
- **Formula:** `new = prev + (target - prev) * smooth`
- **Example:** `prev_screen_x = 500`, `target_x = 1000`, `smooth = 0.2`:
  - Frame 1: 500 + (1000 - 500) \* 0.2 = 600
  - Frame 2: 600 + (1000 - 600) \* 0.2 = 680
  - Frame 3: 680 + (1000 - 680) \* 0.2 = 744
  - The cursor approaches 1000 gradually. Smaller `smooth` = smoother but slower; larger = snappier but jerkier.

---

## 5. Alpha (exponential) smoothing and cursor pipeline

The active code uses **exponential smoothing** (exponential moving average), then **dynamic alpha** and **velocity prediction**. Order in code: target from modified mapping → dynamic alpha → ES → velocity prediction → deadzone check and move.

**Base formula (ES):** `new = alpha * target + (1 - alpha) * prev`. The new position is a blend of the current target and the previous smoothed position; old values fade out exponentially over time.

### 5.1 Dynamic alpha (speed-based)

Alpha is no longer a fixed constant. It is computed each frame from how far the target is from the previous cursor position:

```python
dynamic_x = abs(target_x - prev_screen_x)
dynamic_y = abs(target_y - prev_screen_y)
speed = dynamic_x + dynamic_y
alpha = min(0.35, max(0.08, speed / 1000))
```

- **Idea:** When the hand (and thus the target) has moved a lot, use a larger alpha so the cursor catches up quickly. When the hand is barely moving, use a smaller alpha so the cursor stays smooth and does not jitter.
- **Effect:** Alpha is clamped between 0.08 and 0.35. `speed` is the sum of the x and y pixel distances from the previous position to the target. Dividing by 1000 maps typical movement distances into a reasonable alpha range (e.g. 200 px total → 0.2). Fast motion → higher alpha (more responsive); slow motion → lower alpha (smoother).
- **Example:** If the target jumps 500 px away, `speed = 500`, `speed/1000 = 0.5` → clamped to 0.35. If the target is 50 px away, `speed/1000 = 0.05` → clamped to 0.08.

Then the same ES formula is applied with this alpha:

```python
screen_x = int(alpha * target_x + (1 - alpha) * prev_screen_x)
screen_y = int(alpha * target_y + (1 - alpha) * prev_screen_y)
```

### 5.2 Velocity prediction (reducing micro lag)

After exponential smoothing, the code applies **velocity prediction** so the cursor moves slightly ahead in the direction of motion, reducing perceived lag (comment in code: "velocity prediction, removing micro lag"):

```python
velocity_x = screen_x - prev_screen_x
velocity_y = screen_y - prev_screen_y
screen_x += velocity_x * 0.3
screen_y += velocity_y * 0.3
```

- **Idea:** The change from the previous position to the current smoothed position is treated as a velocity. Adding a fraction of that velocity (0.3) to the position pushes the cursor a bit further in the same direction, so it anticipates the next frame and feels more responsive.
- **Formula:** `final_x = screen_x + (screen_x - prev_screen_x) * 0.3` (and same for y). The factor 0.3 is a tuning value: larger values add more prediction (more snappy, risk of overshoot); smaller values add less (smoother, more lag).
- **Example:** If `prev_screen_x = 100` and smoothed `screen_x = 110`, velocity is 10; then `screen_x` becomes 110 + 10*0.3 = 113. The cursor is placed 3 px ahead in the direction of movement.

The resulting `screen_x` and `screen_y` are what get passed to the deadzone check and `pyautogui.moveTo`.

---

## 6. Deadzone

- **Purpose:** Ignore tiny movements so small hand tremors or noise do not move the cursor.
- **Input:** The position used is the one after modified mapping, dynamic alpha, exponential smoothing, and velocity prediction. Only if that final position is far enough from `prev_screen_x` / `prev_screen_y` does the cursor move.
- **Implementation in code:** A threshold `deadzone = 5` (pixels). The cursor is only updated if the new position differs from the previous by more than 5 pixels in x or y:

```python
if abs(screen_x - prev_screen_x) > deadzone or abs(screen_y - prev_screen_y) > deadzone:
    pyautogui.moveTo(screen_x, screen_y)
    prev_screen_x, prev_screen_y = screen_x, screen_y
```

- **Meaning:** If both `|screen_x - prev_screen_x| <= 5` and `|screen_y - prev_screen_y| <= 5`, the block does nothing and the cursor stays put. Otherwise, the cursor moves and `prev_screen_x` / `prev_screen_y` are updated.
- The move and position update happen only inside this conditional; there is no duplicate afterward, so the deadzone is active and small movements below the threshold are ignored.

---

## 7. Gesture determination (finger up / down)

- Only the four fingers (index, middle, ring, pinky) are used. Tips are landmarks 8, 12, 16, 20; the “knuckle” used for comparison is two joints down: `tip - 2` (e.g. 6, 10, 14, 18).
- **Rule:** Finger is “up” if the tip’s y is **above** the knuckle’s y (smaller y = higher in image):

```python
gesture_determination = [
    1 if hand_landmarks[tip].y < hand_landmarks[tip - 2].y else 0
    for tip in [8, 12, 16, 20]
]
```

- So `gesture_determination` is a list of four 0s and 1s. `sum(gesture_determination) == 4` means all four fingers are up (used for scroll mode).

---

## 8. Click and double-click (thumb–index pinch)

- **Trigger:** Distance between thumb tip (landmark 4) and index tip (landmark 8). `distance = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)` in normalized coordinates.
- **Click when:** `distance < 0.06` (pinch). To avoid repeated clicks while the hand stays pinched, `frozen_cursor` is set to True on pinch and cleared when the hand opens (`distance >= 0.06`). While frozen, cursor movement is disabled but clicks can still be detected.
- **Double-click:** The last two click times are kept. If the latest two clicks are within 0.7 seconds, `pyautogui.doubleClick()` is used and the list is cleared; otherwise a single `pyautogui.click()` is performed. `click_cooldown = 0.5` is defined but the logic uses the 0.7 s window for double-click.

---

## 9. Scroll mode

- **Activation:** When all four fingers are up: `sum(gesture_determination) == 4` sets `scroll_mode = True`; otherwise `scroll_mode = False`.
- **Scroll action:** When in scroll mode and at least 0.2 s since last scroll:
  - Index tip in upper part of frame (`index_tip.y < 0.4`): scroll up, e.g. `pyautogui.scroll(60)`.
  - Index tip in lower part (`index_tip.y > 0.6`): scroll down, e.g. `pyautogui.scroll(-60)`.
- `scroll_time` is used to throttle scrolls so they do not fire every frame.

---

## 10. Skeleton and landmarks (overlay)

- **Skeleton connections:** MediaPipe hand structure is drawn with line segments between landmarks. The `connections` list defines which landmark indices are joined (thumb 0–4, index 0–8, middle 5–12, ring 9–16, pinky 13–20, palm base 0–17). Each segment is drawn with `cv2.line` in pixel coordinates after converting normalized coords with `int(landmark.x * w)` and `int(landmark.y * h)`. Lines are white, thickness 1.
- **Fingertips drawn:** Only fingertips (indices 4, 8, 12, 16, 20) are drawn as circles (radius 6, color (245, 162, 130)) to keep overlay light. Coordinates use the same normalized-to-pixel conversion.

---

## 11. Other important details

- **Camera buffer:** `camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)` sets the capture buffer to one frame. This reduces latency: the program reads the most recent frame instead of an older buffered one, so hand position is closer to real time.
- **PyAutoGUI:** `pyautogui.PAUSE = 0` and `pyautogui.FAILSAFE = False` so the automation does not add delay and does not abort when the mouse hits a screen corner.
- **Status text:** “single click”, “double click”, “scrolling up”, “scrolling down” are shown on the frame for 1 second using `status_time` and `now - status_time < 1`.
- **Frame:** Camera is flipped horizontally (`cv2.flip(frame, 1)`) so movement feels mirror-like. Resolution is set to 640×480.

---

## Summary table

| Concept           | Role                                                                                                   |
| ----------------- | ------------------------------------------------------------------------------------------------------ |
| LERP (commented)    | Linear interpolation: move a fixed fraction of the way to target each frame.                           |
| Dynamic alpha       | Alpha clamped 0.08–0.35 from speed (distance to target); fast motion = snappier, slow = smoother.     |
| Alpha smoothing     | Exponential blend of target and previous position using current alpha.                                |
| Velocity prediction | Add 30% of frame velocity to position to reduce micro lag; final step before deadzone.               |
| Deadzone          | Ignore moves smaller than 5 pixels to reduce jitter; move and prev update only inside the conditional. |
| prev_screen_x / y | Last applied cursor position; used as “previous” for smoothing, velocity, and deadzone.                      |
| frozen_cursor     | Disables cursor movement while thumb–index pinch is held to avoid drift during click.                  |
| Modified mapping    | Offset (-0.1) and scale (*1.25) so hand region maps to full screen and feels less cramped.             |
| Camera buffer = 1   | Lower capture latency by using only one buffered frame.                                                |
