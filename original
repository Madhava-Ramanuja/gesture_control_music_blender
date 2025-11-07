import cv2
import mediapipe as mp
import numpy as np
import pygame
import math

# Initialize pygame mixer
pygame.mixer.init()
# We need two channels, one for each hand
pygame.mixer.set_num_channels(2)

# --- Sound and Channel Setup ---
try:
    sound_right = pygame.mixer.Sound("sounds/Hungry Cheetah.mp3")
    sound_left = pygame.mixer.Sound("sounds/OMG_Daddy.mp3")
except pygame.error as e:
    print(f"Error loading sounds: {e}")
    print("Make sure 'sounds/Hungry Cheetah.mp3' and 'sounds/OMG_Daddy.mp3' exist.")
    exit()

# Create dedicated channels for left and right hands
# Channel 0 for Right Hand, Channel 1 for Left Hand
channel_right = pygame.mixer.Channel(0)
channel_left = pygame.mixer.Channel(1)

# --- MediaPipe Hands ---
mp_hands = mp.solutions.hands
# **MODIFICATION: Set max_num_hands to 2**
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# --- Volume Calculation Constants ---
# NOTE: You may need to adjust these values!
# Run the code and check the console output for your hand's distances.
# This is the average distance from fingertips to wrist.
HAND_OPEN_DIST = 0.6  # Estimated distance for a fully open hand
HAND_CLOSED_DIST = 0.15 # Estimated distance for a closed fist

# Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame
    results = hands.process(rgb_frame)

    # Keep track of which hands are visible in this frame
    hands_seen = {'Left': False, 'Right': False}
    display_info = []

    # Check if any hands are detected
    if results.multi_hand_landmarks:
        # Iterate over ALL detected hands
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            
            # 1. Get Hand Label ("Left" or "Right")
            hand_label = handedness.classification[0].label
            hands_seen[hand_label] = True  # Mark this hand as seen
            
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 2. Calculate Volume based on "Openness"
            lm = hand_landmarks.landmark
            
            # Get coordinates for wrist and fingertips
            wrist_pt = lm[mp_hands.HandLandmark.WRIST]
            index_tip_pt = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip_pt = lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip_pt = lm[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip_pt = lm[mp_hands.HandLandmark.PINKY_TIP]

            # Calculate Euclidean distance from each fingertip to the wrist
            dist_index = math.hypot(index_tip_pt.x - wrist_pt.x, index_tip_pt.y - wrist_pt.y)
            dist_middle = math.hypot(middle_tip_pt.x - wrist_pt.x, middle_tip_pt.y - wrist_pt.y)
            dist_ring = math.hypot(ring_tip_pt.x - wrist_pt.x, ring_tip_pt.y - wrist_pt.y)
            dist_pinky = math.hypot(pinky_tip_pt.x - wrist_pt.x, pinky_tip_pt.y - wrist_pt.y)

            # Average the distances
            avg_dist = (dist_index + dist_middle + dist_ring + dist_pinky) / 4.0

            # print(f"Hand: {hand_label}, Avg Dist: {avg_dist:.2f}")

            # Map the average distance to a volume (0.0 to 1.0)
            volume = np.interp(avg_dist, [HAND_CLOSED_DIST, HAND_OPEN_DIST], [0.0, 1.0])
            volume = max(0.0, min(1.0, volume)) # Clamp the value

            # 3. Control Music on the correct channel
            if hand_label == "Right":
                if not channel_right.get_busy(): # If not playing, start it
                    channel_right.play(sound_right, loops=-1) # Loop forever
                channel_right.set_volume(volume)
            
            elif hand_label == "Left":
                if not channel_left.get_busy(): # If not playing, start it
                    channel_left.play(sound_left, loops=-1) # Loop forever
                channel_left.set_volume(volume)
            
            display_info.append(f"{hand_label} Hand: {int(volume * 100)}%")

    # --- Stop music for hands that are NOT visible ---
    if not hands_seen['Right']:
        channel_right.stop()
    
    if not hands_seen['Left']:
        channel_left.stop()

    # Display the info on the screen
    if not display_info:
        cv2.putText(frame, "No Hands Detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        for i, text in enumerate(display_info):
            # Display info for each hand, stacked vertically
            cv2.putText(frame, text, (10, 50 + (i * 40)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Volume Music Player", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()

