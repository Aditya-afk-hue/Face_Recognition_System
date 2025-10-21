import cv2
import numpy as np
import os

# Ensure the 'images' directory exists
if not os.path.exists('images'):
    os.makedirs('images')

# Load the Haar Cascade classifier for face detection
classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize video capture from the default webcam (index 0)
cap = cv2.VideoCapture(0)

data = []

while len(data) < 100:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Detect faces in the frame
    face_points = classifier.detectMultiScale(frame, 1.3, 5)

    if len(face_points) > 0:
        for x, y, w, h in face_points:
            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Crop the face from the frame
            face_frame = frame[y:y+h, x:x+w]
            
            # Show only the cropped face in a separate window
            cv2.imshow("Only face", face_frame)

            # Collect 100 face samples
            if len(data) < 100:
                print(len(data) + 1, "/100")
                data.append(face_frame)
            else:
                break
    
    # Display the number of collected images on the main frame
    cv2.putText(frame, f"{len(data)}/100", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("frame", frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()

# Save the collected data if 100 samples were captured
if len(data) == 100:
    name = input("Enter the person's name: ")
    for i in range(100):
        # Save each face image to the 'images' directory
        cv2.imwrite(f"images/{name}_{i}.jpg", data[i])
    print("Data collection complete!")
else:
    print("Could not collect 100 images. Please try again.")