import cv2
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder # ---> Need LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- 1. Data Loading and Preparation ---
print("Loading images and labels...")
img_dir = 'images'
image_data = []
labels_list = []
label_map = {}
current_label_id = 0

# Check if the images directory exists
if not os.path.exists(img_dir):
    print(f"Error: Directory '{img_dir}' not found. Please run collect_data.py first.")
    exit()

for filename in os.listdir(img_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"): # Check for image files
        try:
            name = filename.split("_")[0]
            if name not in label_map:
                label_map[name] = current_label_id
                current_label_id += 1

            label_id = label_map[name]

            image_path = os.path.join(img_dir, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image {image_path}. Skipping.")
                continue

            # Preprocess the image (same as in recognize.py's preprocess without reshape/normalize)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (100, 100))
            image = cv2.equalizeHist(image) # Apply histogram equalization

            image_data.append(image)
            labels_list.append(label_id)
        except Exception as e:
            print(f"Error processing file {filename}: {e}")

if not image_data:
    print("Error: No images found or loaded. Did you collect data?")
    exit()

# Convert to NumPy arrays
image_data = np.array(image_data)
labels_array = np.array(labels_list)

# Reshape image data for CNN (add channel dimension) and normalize
image_data = image_data.reshape(image_data.shape[0], 100, 100, 1)
image_data = image_data / 255.0

# Encode labels to integers and then to categorical format
encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels_array)
y = to_categorical(labels_encoded)
num_classes = len(np.unique(labels_encoded))

# Reverse map for saving labels.npy correctly {index: name}
# Create the reverse map {id: name}
id_to_name_map = {v: k for k, v in label_map.items()}
np.save('labels.npy', id_to_name_map)
print("Saved label map to labels.npy:", id_to_name_map)


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(image_data, y, test_size=0.2, random_state=42, stratify=y)
# ---> END OF DATA LOADING AND PREPARATION <---


print(f"Number of classes found: {num_classes}")
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")


# ---> ADD DATA AUGMENTATION SETUP <---
print("Setting up data augmentation...")
datagen = ImageDataGenerator(
    rotation_range=10,      # Rrotate images slightly
    width_shift_range=0.1,  # Shift images horizontally
    height_shift_range=0.1, # Shift images vertically
    shear_range=0.1,        # Apply shear transformation
    zoom_range=0.1,         # Zoom in/out slightly
    horizontal_flip=False,  # Usually not needed for faces unless profile view
    fill_mode='nearest'
)
datagen.fit(X_train)
# ---> END OF AUGMENTATION SETUP <---


# --- 2. Build the CNN Model ---
print("Building the model...")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5), # Dropout for regularization
    Dense(num_classes, activation='softmax') # Output layer size = number of people
])
model.summary()


# --- 3. Compile and Train the Model ---
print("Compiling and training the model...")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ---> MODIFY model.fit TO USE AUGMENTATION <---
print("Training with augmented data...")
# Increase epochs for better learning with augmentation
epochs_to_train = 30 # Increased from 20
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    epochs=epochs_to_train,
                    validation_data=(X_test, y_test),
                    steps_per_epoch=max(1, len(X_train) // 32)) # Ensure steps_per_epoch is at least 1
# ---> END OF MODIFICATION <---


# --- 4. Evaluate the Model (Optional but Recommended) ---
print("Evaluating model...")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")


# --- 5. Save the Model ---
print("Saving the trained model...")
model.save("final_model.h5")
# Label map was saved earlier after encoding
print("Training complete! Your model 'final_model.h5' and 'labels.npy' are ready.")