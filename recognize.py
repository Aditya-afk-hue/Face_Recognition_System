import cv2
import numpy as np
from keras.models import load_model
# ---> ADD THIS IMPORT <---
from collections import deque 

# Load the Haar Cascade classifier for face detection
classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load your pre-trained model and the label map
model = load_model("final_model.h5")
labels = np.load('labels.npy', allow_pickle=True).item()

# Initialize video capture from the default webcam (index 0)
cap = cv2.VideoCapture(0)

# ---> ADD VARIABLES FOR SMOOTHING <---
prediction_history = {} # Dictionary to store recent predictions for each face ID
SMOOTHING_WINDOW_SIZE = 10 # Number of frames to average over
CONFIDENCE_THRESHOLD = 0.70 # Minimum confidence needed to accept prediction (adjust as needed)
next_face_id = 0
face_tracker = {} # Dictionary to track faces across frames (optional but improves tracking)
# ---> END OF ADDED VARIABLES <---


def get_pred_label(prediction_probs):
    prediction_index = np.argmax(prediction_probs)
    confidence = prediction_probs[0][prediction_index] # Get the confidence score
    
    # Check if confidence meets the threshold
    if confidence >= CONFIDENCE_THRESHOLD:
        return labels.get(prediction_index, "Unknown"), confidence
    else:
        return "Unknown", confidence # Return "Unknown" if not confident enough

def preprocess(img):
    # ... (Keep your preprocessing function the same) ...
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (100, 100))
    img = cv2.equalizeHist(img)
    img = img.reshape(1, 100, 100, 1)
    img = img / 255.0
    return img

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    faces = classifier.detectMultiScale(frame, 1.5, 5)

    current_face_centers = []
    for x, y, w, h in faces:
        face_center = (x + w // 2, y + h // 2)
        current_face_centers.append(((x, y, w, h), face_center))

    # Basic tracking logic (can be improved with dedicated trackers)
    matched_face_ids = set()
    new_face_tracker = {}

    for (x, y, w, h), center in current_face_centers:
        best_match_id = -1
        min_dist = 100 # Max distance to consider a match

        # Find the closest tracked face from the previous frame
        for face_id, last_center in face_tracker.items():
            dist = np.linalg.norm(np.array(center) - np.array(last_center))
            if dist < min_dist:
                min_dist = dist
                best_match_id = face_id

        if best_match_id != -1 and best_match_id not in matched_face_ids:
            # Matched an existing face
            face_id = best_match_id
            matched_face_ids.add(face_id)
        else:
            # New face detected
            face_id = next_face_id
            next_face_id += 1
            prediction_history[face_id] = deque(maxlen=SMOOTHING_WINDOW_SIZE) # Initialize history for new face

        new_face_tracker[face_id] = center # Update tracker with current center

        # Crop, preprocess, and predict
        face = frame[y:y+h, x:x+w]
        processed_face = preprocess(face)
        prediction_probs = model.predict(processed_face)
        label, confidence = get_pred_label(prediction_probs)

        # ---> APPLY SMOOTHING <---
        # Add current prediction to this face's history
        history = prediction_history.setdefault(face_id, deque(maxlen=SMOOTHING_WINDOW_SIZE))
        history.append(label)

        # Determine the most frequent label in the history
        if history:
            most_common_label = max(set(history), key=list(history).count)
            # Only display if the most common is not "Unknown" or if it's the only prediction
            if most_common_label != "Unknown" or len(set(history)) == 1:
                display_label = most_common_label
            else: 
                # If "Unknown" is frequent, try the second most common if available
                counts = {l: list(history).count(l) for l in set(history)}
                sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
                if len(sorted_counts) > 1 and sorted_counts[1][0] != "Unknown":
                    display_label = sorted_counts[1][0] # Display second most common if not Unknown
                else:
                    display_label = "Unknown" # Otherwise default to Unknown
        else:
            display_label = "Unknown"
        # ---> END OF SMOOTHING <---

        # Display results
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Display the smoothed label and confidence
        display_text = f"{display_label} ({confidence:.2f})" 
        cv2.putText(frame, display_text, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)

    face_tracker = new_face_tracker # Update the tracker for the next frame


    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()