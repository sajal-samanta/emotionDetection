import tensorflow as tf
import cv2
import numpy as np
import json
import os

print("🔍 Testing Model Loading...")

# Check if model files exist
model_files = ['model/emotion_model.h5', 'model/emotion_model.keras', 'model/emotion_labels.json']
for file in model_files:
    if os.path.exists(file):
        print(f"✅ Found: {file}")
    else:
        print(f"❌ Missing: {file}")

# Try to load the model
try:
    if os.path.exists('model/emotion_model.keras'):
        model = tf.keras.models.load_model('model/emotion_model.keras')
        print("✅ Model loaded successfully from .keras file")
    elif os.path.exists('model/emotion_model.h5'):
        model = tf.keras.models.load_model('model/emotion_model.h5')
        print("✅ Model loaded successfully from .h5 file")
    else:
        print("❌ No model file found")
        model = None
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Try to load emotion labels
try:
    with open('model/emotion_labels.json', 'r') as f:
        emotion_labels = json.load(f)
    print("✅ Emotion labels loaded:", emotion_labels)
except Exception as e:
    print(f"❌ Error loading emotion labels: {e}")

# Test face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("❌ Face cascade classifier not loaded!")
else:
    print("✅ Face cascade classifier loaded")

print("🎯 Test completed!")