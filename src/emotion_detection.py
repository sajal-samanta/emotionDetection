import cv2
import streamlit as st
import numpy as np
from utils import EmotionDetector
import time

class WebcamEmotionDetection:
    def __init__(self, model_path='../model/emotion_model.h5'):
        self.emotion_labels = {
            0: 'Angry',
            1: 'Disgust', 
            2: 'Fear',
            3: 'Happy',
            4: 'Sad',
            5: 'Surprise',
            6: 'Neutral'
        }
        self.detector = EmotionDetector(model_path, self.emotion_labels)
        self.is_running = False
    
    def start_webcam(self):
        """Start webcam emotion detection"""
        stframe = st.empty()
        stop_button = st.button('Stop Webcam')
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.is_running = True
        emotion_stats = {emotion: 0 for emotion in self.emotion_labels.values()}
        
        while self.is_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image from webcam")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect faces and emotions
            faces, gray = self.detector.detect_faces(frame)
            
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                emotion, confidence, all_predictions = self.detector.predict_emotion(face_roi)
                
                # Update statistics
                emotion_stats[emotion] += 1
                
                # Draw prediction on frame
                frame = self.detector.draw_prediction(frame, x, y, w, h, emotion, confidence)
                
                # Draw emotion probabilities
                self.draw_emotion_probabilities(frame, all_predictions)
            
            # Convert BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels='RGB', use_column_width=True)
            
            if stop_button:
                self.is_running = False
                break
        
        cap.release()
        return emotion_stats
    
    def draw_emotion_probabilities(self, image, predictions):
        """Draw emotion probabilities on the side of the image"""
        y_offset = 30
        for i, (emotion, prob) in enumerate(zip(self.emotion_labels.values(), predictions)):
            color = self.detector.get_emotion_color(emotion)
            text = f"{emotion}: {prob:.2f}"
            cv2.putText(image, text, (10, y_offset + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)