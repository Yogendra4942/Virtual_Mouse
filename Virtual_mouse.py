import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Get screen size
screen_width, screen_height = pyautogui.size()

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip for better control
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get index finger tip coordinates
            index_finger_tip = hand_landmarks.landmark[8]
            x = int(index_finger_tip.x * screen_width)
            y = int(index_finger_tip.y * screen_height)

            # Move cursor
            pyautogui.moveTo(x, y, duration=0.1)

            # Detect click gesture (thumb and index close)
            thumb_tip = hand_landmarks.landmark[4]
            distance = np.linalg.norm(
                np.array([index_finger_tip.x, index_finger_tip.y]) -
                np.array([thumb_tip.x, thumb_tip.y])
            )

            if distance < 0.02:  # Click when thumb and index finger are close
                pyautogui.click()

    cv2.imshow("AI Virtual Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
