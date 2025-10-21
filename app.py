import cv2
import numpy as np
from keras.models import load_model
from collections import deque
import base64
from flask import Flask, render_template, request # <-- CHANGED (added request)
from flask_socketio import SocketIO, emit
import os
from PIL import Image
import io
import eventlet # <-- CHANGED (Import eventlet at the top)

# --- Initialization ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key!' # Change this!
socketio = SocketIO(app, async_mode='eventlet')

print("Loading resources...")
# Load models and classifiers ONCE at startup
try:
    classifier_path = 'haarcascade_frontalface_default.xml'
    model_path = 'final_model.h5'
    labels_path = 'labels.npy'

    if not os.path.exists(classifier_path):
        raise FileNotFoundError(f"Haar Cascade not found at {classifier_path}")
    if not os.path.exists(model_path):
         raise FileNotFoundError(f"Model not found at {model_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found at {labels_path}")

    classifier = cv2.CascadeClassifier(classifier_path)
    model = load_model(model_path)
    labels = np.load(labels_path, allow_pickle=True).item()
    print("Resources loaded successfully.")
except Exception as e:
    print(f"Error loading resources: {e}")
    exit()


# --- Face Recognition Logic (Adapted from your recognize.py) ---
# Global state for smoothing
SMOOTHING_WINDOW_SIZE = 10
CONFIDENCE_THRESHOLD = 0.70

# --- CHANGED: Removed global state. Will be stored in sessions ---
# prediction_history = {}
# next_face_id = 0
# face_tracker = {}
sessions = {} # <-- CHANGED: Main dictionary to hold all session data

def preprocess(img):
    # Keep preprocessing consistent with training
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Input is likely BGR
    img_gray = cv2.resize(img_gray, (100, 100))
    img_gray = cv2.equalizeHist(img_gray)
    img_res = img_gray.reshape(1, 100, 100, 1)
    img_res = img_res / 255.0
    return img_res

def get_pred_label(prediction_probs):
    prediction_index = np.argmax(prediction_probs)
    confidence = prediction_probs[0][prediction_index]
    if confidence >= CONFIDENCE_THRESHOLD:
        return labels.get(prediction_index, "Unknown"), confidence
    else:
        return "Unknown", confidence

# <-- CHANGED: Function now accepts session_state
def process_frame(frame, session_state): 
    # <-- CHANGED: Get state from the session dictionary
    next_face_id = session_state['next_face_id']
    face_tracker = session_state['face_tracker']
    prediction_history = session_state['prediction_history']
    
    # Detect faces
    faces = classifier.detectMultiScale(frame, 1.3, 5) 

    current_face_centers = []
    for (x, y, w, h) in faces:
        face_center = (x + w // 2, y + h // 2)
        current_face_centers.append(((x, y, w, h), face_center))

    # Basic tracking logic
    matched_face_ids = set()
    new_face_tracker = {}

    for (x, y, w, h), center in current_face_centers:
        best_match_id = -1
        min_dist = 100 

        for face_id, last_center in face_tracker.items():
            dist = np.linalg.norm(np.array(center) - np.array(last_center))
            if dist < min_dist:
                min_dist = dist
                best_match_id = face_id

        if best_match_id != -1 and best_match_id not in matched_face_ids:
            face_id = best_match_id
            matched_face_ids.add(face_id)
        else:
            face_id = next_face_id
            next_face_id += 1 # <-- This updates the local variable
            prediction_history[face_id] = deque(maxlen=SMOOTHING_WINDOW_SIZE)

        new_face_tracker[face_id] = center

        # Crop, preprocess, predict
        face_roi = frame[y:y+h, x:x+w]
        if face_roi.size == 0: continue

        processed_face = preprocess(face_roi) 
        prediction_probs = model.predict(processed_face, verbose=0) 
        label, confidence = get_pred_label(prediction_probs)

        # --- Smoothing ---
        history = prediction_history.setdefault(face_id, deque(maxlen=SMOOTHING_WINDOW_SIZE))
        history.append(label)

        display_label = "Processing..."
        if history:
            valid_labels = [l for l in history if l != "Unknown"]
            if valid_labels:
                most_common_label = max(set(valid_labels), key=valid_labels.count)
                if valid_labels.count(most_common_label) > SMOOTHING_WINDOW_SIZE // 2:
                     display_label = most_common_label
                else:
                     display_label = "Processing..."
            else:
                display_label = "Unknown"
        # --- End Smoothing ---

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        display_text = f"{display_label} ({confidence:.2f})"
        cv2.putText(frame, display_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    face_tracker = new_face_tracker # Update local tracker

    # Cleanup old entries if tracker gets too large
    if len(face_tracker) > 50:
         face_tracker = {k: v for k, v in face_tracker.items() if k in new_face_tracker}
         prediction_history = {k: v for k, v in prediction_history.items() if k in new_face_tracker}

    # <-- CHANGED: Save updated state back to the session
    session_state['next_face_id'] = next_face_id
    session_state['face_tracker'] = face_tracker
    session_state['prediction_history'] = prediction_history

    return frame


# --- Flask Routes ---
@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

# --- SocketIO Events ---
@socketio.on('connect')
def handle_connect():
    # <-- CHANGED: Create a new state for each connecting client
    print(f'Client connected: {request.sid}')
    sessions[request.sid] = {
        'prediction_history': {},
        'next_face_id': 0,
        'face_tracker': {}
    }

@socketio.on('disconnect')
def handle_disconnect():
    # <-- CHANGED: Clean up the state when a client disconnects
    print(f'Client disconnected: {request.sid}')
    if request.sid in sessions:
        del sessions[request.sid]

@socketio.on('image')
def handle_image(data_image):
    # <-- CHANGED: Get the specific state for this client
    session_state = sessions.get(request.sid)
    if not session_state:
        # This can happen if server restarts or connection glitches
        print(f"Warning: No state for {request.sid}, re-initializing.")
        handle_connect() # Re-create the state
        session_state = sessions[request.sid]

    # Decode base64 image data
    try:
        sbuf = io.BytesIO()
        sbuf.write(base64.b64decode(data_image.split(',')[1])) 
        pimg = Image.open(sbuf)
        frame = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

        # <-- CHANGED: Pass the client's state to be processed
        processed_frame = process_frame(frame, session_state)

        # Encode processed frame back to JPEG and then base64
        _, buffer = cv2.imencode('.jpg', processed_frame)
        b64_frame = base64.b64encode(buffer).decode('utf-8')

        # Send it back to the client
        emit('response_back', f'data:image/jpeg;base64,{b64_frame}')
    except Exception as e:
        print(f"Error processing image: {e}")


if __name__ == '__main__':
    print("Starting server...")
    # --- CHANGED: Make sure you are binding to 0.0.0.0 ---
    # Your log showed 127.0.0.1, which is also fine for local testing
    # But 0.0.0.0 is more flexible.

    # This is what Render requires:
    eventlet.wsgi.server(eventlet.listen(('0.0.0.0', 5000)), app)
