import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import os

print("🔄 Loading Emotion Detector...")

class EmotionDetector:
    def __init__(self, model_path='model/emotion_model.h5', labels_path='model/emotion_labels.json'):
        self.model = None
        self.face_cascade = None
        self.emotion_labels = {
            0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 
            4: 'Sad', 5: 'Surprise', 6: 'Neutral'
        }
        
        try:
            print("🔧 Initializing Emotion Detector...")
            
            # Load face detector
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if self.face_cascade.empty():
                print("❌ Failed to load face cascade classifier")
            else:
                print("✅ Face cascade classifier loaded")
            
            # Try different model paths
            possible_paths = [
                model_path,
                'model/emotion_model.keras',
                'model/emotion_model.h5',
                '../model/emotion_model.keras',
                '../model/emotion_model.h5'
            ]
            
            model_loaded = False
            for path in possible_paths:
                if os.path.exists(path):
                    print(f"📁 Trying to load model from: {path}")
                    try:
                        self.model = load_model(path)
                        print(f"✅ Model loaded successfully from: {path}")
                        model_loaded = True
                        break
                    except Exception as e:
                        print(f"❌ Failed to load from {path}: {e}")
            
            if not model_loaded:
                print("❌ Could not load any model file")
                self.model = None
            
            # Load emotion labels
            possible_label_paths = [
                labels_path,
                'model/emotion_labels.json',
                '../model/emotion_labels.json'
            ]
            
            labels_loaded = False
            for path in possible_label_paths:
                if os.path.exists(path):
                    try:
                        with open(path, 'r') as f:
                            loaded_labels = json.load(f)
                        # Convert string keys to integers
                        self.emotion_labels = {int(k): v for k, v in loaded_labels.items()}
                        print(f"✅ Emotion labels loaded from: {path}")
                        labels_loaded = True
                        break
                    except Exception as e:
                        print(f"❌ Failed to load labels from {path}: {e}")
            
            if not labels_loaded:
                print("⚠️ Using default emotion labels")
            
            print(f"🎭 Available emotions: {list(self.emotion_labels.values())}")
            
        except Exception as e:
            print(f"❌ Error initializing emotion detector: {e}")
    
    def detect_faces(self, image):
        """Detect faces in the image"""
        if self.face_cascade is None or self.face_cascade.empty():
            # Return dummy face for testing
            h, w = image.shape[:2]
            return [(w//4, h//4, w//2, h//2)], cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces, gray
    
    def preprocess_face(self, face_roi):
        """Preprocess face ROI for emotion prediction"""
        # Resize to 48x48 (FER2013 input size)
        face_roi = cv2.resize(face_roi, (48, 48))
        # Normalize
        face_roi = face_roi.astype('float32') / 255.0
        # Add channel dimension
        face_roi = np.expand_dims(face_roi, axis=-1)
        # Add batch dimension
        face_roi = np.expand_dims(face_roi, axis=0)
        return face_roi
    
    def predict_emotion(self, face_roi):
        """Predict emotion from face ROI"""
        if self.model is None:
            # Return mock prediction for testing
            print("⚠️ Using mock prediction (no model)")
            return "Happy", 0.95, [0.1, 0.1, 0.1, 0.6, 0.05, 0.05, 0.0]
        
        try:
            processed_face = self.preprocess_face(face_roi)
            predictions = self.model.predict(processed_face, verbose=0)
            emotion_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            # Debug output
            print(f"🎭 Predicted: {self.emotion_labels[emotion_idx]} (Confidence: {confidence:.2f})")
            
            return self.emotion_labels[emotion_idx], confidence, predictions[0]
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return "Neutral", 0.5, [0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14]
    
    def draw_prediction(self, image, x, y, w, h, emotion, confidence):
        """Draw bounding box and emotion prediction on image"""
        color = self.get_emotion_color(emotion)
        
        # Draw bounding box
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        
        # Draw emotion label with confidence
        label = f"{emotion}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # Draw background for text
        cv2.rectangle(image, (x, y - label_size[1] - 10), 
                     (x + label_size[0], y), color, -1)
        
        # Draw text
        cv2.putText(image, label, (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return image
    
    def get_emotion_color(self, emotion):
        """Get color based on emotion"""
        colors = {
            'Angry': (0, 0, 255),      # Red
            'Disgust': (0, 128, 0),    # Green
            'Fear': (128, 0, 128),     # Purple
            'Happy': (0, 255, 255),    # Yellow
            'Sad': (255, 0, 0),        # Blue
            'Surprise': (0, 165, 255), # Orange
            'Neutral': (128, 128, 128) # Gray
        }
        return colors.get(emotion, (255, 255, 255))

print("✅ Emotion Detector class defined")