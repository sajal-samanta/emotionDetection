import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import sys
import time
import pandas as pd
from datetime import datetime, timedelta
import json
import hashlib

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.utils import EmotionDetector
except ImportError:
    st.error("❌ Could not import EmotionDetector. Make sure utils.py is in src/ folder")

# Page configuration
st.set_page_config(
    page_title="AI Emotion Detection",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (keep your existing CSS here)
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .emotion-stats {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .analytics-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .profile-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
    }
    .viz-controls {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class UserProfileManager:
    """Manage user profiles and emotion history"""
    
    def __init__(self):
        self.profiles_dir = "user_profiles"
        os.makedirs(self.profiles_dir, exist_ok=True)
    
    def create_user_profile(self, username, password):
        """Create a new user profile"""
        user_id = hashlib.md5(f"{username}{password}".encode()).hexdigest()
        profile_path = os.path.join(self.profiles_dir, f"{user_id}.json")
        
        if os.path.exists(profile_path):
            return False, "User already exists"
        
        profile = {
            'username': username,
            'user_id': user_id,
            'created_at': datetime.now().isoformat(),
            'emotion_history': [],
            'sessions': [],
            'preferences': {
                'theme': 'light',
                'viz_style': 'default',
                'privacy_level': 'private'
            }
        }
        
        with open(profile_path, 'w') as f:
            json.dump(profile, f, indent=2)
        
        return True, user_id
    
    def login_user(self, username, password):
        """Login existing user"""
        user_id = hashlib.md5(f"{username}{password}".encode()).hexdigest()
        profile_path = os.path.join(self.profiles_dir, f"{user_id}.json")
        
        if os.path.exists(profile_path):
            with open(profile_path, 'r') as f:
                profile = json.load(f)
            return True, profile
        return False, None
    
    def convert_to_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    def update_user_profile(self, user_id, emotion_data, session_data=None):
        """Update user profile with new emotion data"""
        profile_path = os.path.join(self.profiles_dir, f"{user_id}.json")
        
        if os.path.exists(profile_path):
            with open(profile_path, 'r') as f:
                profile = json.load(f)
            
            # Convert emotion data to serializable types
            serializable_emotion_data = self.convert_to_serializable(emotion_data)
            
            # Add emotion to history
            emotion_entry = {
                'emotion': serializable_emotion_data['emotion'],
                'confidence': serializable_emotion_data['confidence'],
                'timestamp': datetime.now().isoformat(),
                'session_id': session_data['session_id'] if session_data else 'anonymous'
            }
            profile['emotion_history'].append(emotion_entry)
            
            # Keep only last 1000 entries
            if len(profile['emotion_history']) > 1000:
                profile['emotion_history'] = profile['emotion_history'][-1000:]
            
            # Update session data if provided
            if session_data and session_data not in profile['sessions']:
                profile['sessions'].append(session_data)
            
            with open(profile_path, 'w') as f:
                json.dump(profile, f, indent=2)
            
            return True
        return False
    
    def get_user_stats(self, user_id):
        """Get comprehensive user statistics"""
        profile_path = os.path.join(self.profiles_dir, f"{user_id}.json")
        
        if os.path.exists(profile_path):
            with open(profile_path, 'r') as f:
                profile = json.load(f)
            
            emotion_history = profile['emotion_history']
            if not emotion_history:
                return None
            
            # Calculate statistics
            df = pd.DataFrame(emotion_history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            stats = {
                'total_sessions': len(profile['sessions']),
                'total_emotions': len(emotion_history),
                'dominant_emotion': df['emotion'].mode().iloc[0] if not df.empty else 'Neutral',
                'avg_confidence': float(df['confidence'].mean()) if not df.empty else 0.0,
                'emotion_distribution': df['emotion'].value_counts().to_dict(),
                'recent_activity': len([e for e in emotion_history 
                                      if pd.to_datetime(e['timestamp']) > datetime.now() - timedelta(days=7)]),
                'happiness_ratio': float(len(df[df['emotion'] == 'Happy']) / len(df) * 100) if not df.empty else 0.0
            }
            
            return stats
        return None

class EmotionDetectionApp:
    def __init__(self):
        # Initialize session state variables
        self.initialize_session_state()
        
        try:
            self.detector = EmotionDetector()
            self.emotion_stats = {emotion: 0 for emotion in self.detector.emotion_labels.values()}
            # Initialize analytics data
            self.emotion_history = []
            self.session_start_time = None
            self.current_emotion = "Neutral"
            self.current_confidence = 0.0
            self.frame_count = 0
            # Initialize user profile manager
            self.profile_manager = UserProfileManager()
            # Visualization settings
            self.viz_settings = {
                'chart_style': 'default',
                'show_3d': False,
                'animate_transitions': True,
                'color_scheme': 'vibrant'
            }
        except Exception as e:
            st.error(f"❌ Error initializing emotion detector: {e}")
            self.detector = None
            self.emotion_stats = {}
            self.emotion_history = []
    
    def initialize_session_state(self):
        """Initialize all session state variables"""
        if 'user_logged_in' not in st.session_state:
            st.session_state.user_logged_in = False
        if 'username' not in st.session_state:
            st.session_state.username = None
        if 'user_id' not in st.session_state:
            st.session_state.user_id = None
        if 'user_stats' not in st.session_state:
            st.session_state.user_stats = None
        if 'webcam_running' not in st.session_state:
            st.session_state.webcam_running = False
    
    def render_sidebar(self):
        st.sidebar.title("🎯 AI Emotion Detection")
        st.sidebar.markdown("---")
        
        # User Profile Section
        if st.session_state.user_logged_in:
            st.sidebar.markdown("""
            <div class="profile-card">
                <h3>👤 User Profile</h3>
                <p><strong>{}</strong></p>
                <p>{} emotions tracked</p>
            </div>
            """.format(
                st.session_state.username,
                st.session_state.user_stats['total_emotions'] if st.session_state.user_stats else 0
            ), unsafe_allow_html=True)
            
            if st.sidebar.button('🚪 Logout', use_container_width=True):
                st.session_state.user_logged_in = False
                st.session_state.username = None
                st.session_state.user_id = None
                st.session_state.user_stats = None
                st.rerun()
        else:
            st.sidebar.subheader("🔐 User Login")
            with st.sidebar.expander("Login / Register"):
                tab1, tab2 = st.tabs(["Login", "Register"])
                
                with tab1:
                    login_username = st.text_input("Username", key="login_user")
                    login_password = st.text_input("Password", type="password", key="login_pass")
                    if st.button("Login", key="login_btn"):
                        success, profile = self.profile_manager.login_user(login_username, login_password)
                        if success:
                            st.session_state.user_logged_in = True
                            st.session_state.username = login_username
                            st.session_state.user_id = profile['user_id']
                            st.session_state.user_stats = self.profile_manager.get_user_stats(profile['user_id'])
                            st.success("✅ Login successful!")
                            st.rerun()
                        else:
                            st.error("❌ Invalid credentials")
                
                with tab2:
                    reg_username = st.text_input("Username", key="reg_user")
                    reg_password = st.text_input("Password", type="password", key="reg_pass")
                    if st.button("Register", key="reg_btn"):
                        success, user_id = self.profile_manager.create_user_profile(reg_username, reg_password)
                        if success:
                            st.session_state.user_logged_in = True
                            st.session_state.username = reg_username
                            st.session_state.user_id = user_id
                            st.session_state.user_stats = self.profile_manager.get_user_stats(user_id)
                            st.success("✅ Registration successful!")
                            st.rerun()
                        else:
                            st.error("❌ User already exists")
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Project Info")
        st.sidebar.info("""
        Real-time facial emotion recognition using:
        - Deep Learning (CNN)
        - Computer Vision (OpenCV)
        - Real-time Webcam
        - Streamlit Web Interface
        """)
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎭 Detected Emotions")
        for emotion, color in [
            ('😠 Angry', '#FF0000'),
            ('🤢 Disgust', '#008000'), 
            ('😨 Fear', '#800080'),
            ('😄 Happy', '#FFFF00'),
            ('😢 Sad', '#0000FF'),
            ('😲 Surprise', '#FFA500'),
            ('😐 Neutral', '#808080')
        ]:
            st.sidebar.markdown(f"<span style='color:{color}; font-weight:bold;'>{emotion}</span>", 
                              unsafe_allow_html=True)
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("🛠️ Model Info")
        st.sidebar.success("""
        ✅ Model Trained!
        📊 Accuracy: ~64%
        🎯 7 Emotions
        ⚡ Ready for Inference
        """)
    
    def render_visualization_controls(self):
        """Render advanced visualization controls"""
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎨 Visualization Settings")
        
        with st.sidebar.expander("Customize Visualizations"):
            # Chart style selection
            self.viz_settings['chart_style'] = st.selectbox(
                "Chart Style",
                ["default", "minimal", "dark", "seaborn", "plotly_white"],
                index=0
            )
            
            # Color scheme
            self.viz_settings['color_scheme'] = st.selectbox(
                "Color Scheme",
                ["vibrant", "pastel", "monochrome", "warm", "cool"],
                index=0
            )
            
            # 3D visualization toggle
            self.viz_settings['show_3d'] = st.checkbox("Show 3D Visualizations", value=False)
            
            # Animation toggle
            self.viz_settings['animate_transitions'] = st.checkbox("Animate Transitions", value=True)
            
            if st.button("Apply Settings"):
                st.success("✅ Visualization settings updated!")
    
    def apply_viz_theme(self, fig):
        """Apply visualization theme to plotly figure"""
        color_schemes = {
            'vibrant': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8'],
            'pastel': ['#FFB3BA', '#BAFFC9', '#BAE1FF', '#FFFFBA', '#FFDFBA', '#E0BBE4', '#B5EAD7'],
            'monochrome': ['#666666', '#888888', '#AAAAAA', '#CCCCCC', '#EEEEEE', '#F5F5F5', '#FAFAFA'],
            'warm': ['#FF6B6B', '#FFA726', '#FFCA28', '#FFE082', '#FFF59D', '#FFECB3', '#FFE0B2'],
            'cool': ['#42A5F5', '#5C6BC0', '#26C6DA', '#26A69A', '#66BB6A', '#9CCC65', '#D4E157']
        }
        
        # Apply color scheme
        colors = color_schemes.get(self.viz_settings['color_scheme'], color_schemes['vibrant'])
        
        # Update figure colors
        if hasattr(fig, 'data'):
            for i, trace in enumerate(fig.data):
                if hasattr(trace, 'marker') and hasattr(trace.marker, 'color'):
                    if isinstance(trace.marker.color, list):
                        trace.marker.color = colors
                if hasattr(trace, 'line') and hasattr(trace.line, 'color'):
                    trace.line.color = colors[i % len(colors)]
        
        # Apply chart style
        if self.viz_settings['chart_style'] == 'dark':
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
        elif self.viz_settings['chart_style'] == 'minimal':
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                margin=dict(l=20, r=20, t=30, b=20)
            )
        
        return fig

    def render_home(self):
        st.markdown('<div class="main-header">🤖 AI-Powered Face Emotion Detection</div>', 
                   unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="feature-box">
            <h3>🎯 What This Project Demonstrates</h3>
            <ul>
                <li><strong>Deep Learning</strong>: Custom CNN architecture for emotion classification</li>
                <li><strong>Real-time Inference</strong>: Live webcam emotion detection</li>
                <li><strong>Computer Vision</strong>: Face detection and preprocessing with OpenCV</li>
                <li><strong>Web Deployment</strong>: Interactive Streamlit web application</li>
                <li><strong>Model Training</strong>: 63.9% accuracy on FER2013 dataset</li>
                <li><strong>Real-time Analytics</strong>: Live emotion tracking and insights</li>
                <li><strong>Personal Profiles</strong>: User accounts with emotion history</li>
                <li><strong>Advanced Visualizations</strong>: 3D charts and interactive graphs</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="emotion-stats">
            <h3>🚀 Live Features</h3>
            <p>📷 Real-time Webcam Detection</p>
            <p>📊 Live Emotion Analytics</p>
            <p>👤 Personal Emotion Profiles</p>
            <p>🎨 Advanced Visualizations</p>
            <p>🖼️ Image Upload & Analysis</p>
            <p>📈 Emotion Statistics</p>
            <p>🎯 Multi-face Detection</p>
            <p>⚡ Fast Inference</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Quick stats
        st.subheader("📈 Model Performance")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Validation Accuracy", "63.9%")
        with col2:
            st.metric("Emotions Detected", "7")
        with col3:
            st.metric("Training Epochs", "56")
        with col4:
            st.metric("Model Ready", "✅")

    def render_real_time_analytics(self):
        """Render real-time emotion analytics dashboard"""
        st.subheader("📊 Live Emotion Analytics")
        
        if not self.emotion_history:
            st.info("🔍 Start webcam to see real-time analytics")
            return
        
        # Calculate analytics
        total_detections = sum(self.emotion_stats.values())
        if total_detections == 0:
            st.info("No emotions detected yet")
            return
        
        # Create columns for analytics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Dominant Emotion
            dominant_emotion = max(self.emotion_stats, key=self.emotion_stats.get)
            dominant_count = self.emotion_stats[dominant_emotion]
            dominant_percentage = (dominant_count / total_detections) * 100
            
            st.markdown(f"""
            <div class="analytics-card">
                <h3>🏆 Dominant Emotion</h3>
                <h2 style="color: #1f77b4; margin: 0.5rem 0;">{dominant_emotion}</h2>
                <p>{dominant_percentage:.1f}% of detections</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Happiness Index
            happy_count = self.emotion_stats.get('Happy', 0)
            happiness_ratio = (happy_count / total_detections) * 100
            
            st.markdown(f"""
            <div class="analytics-card">
                <h3>😊 Happiness Index</h3>
                <h2 style="color: #FF6B6B; margin: 0.5rem 0;">{happiness_ratio:.1f}%</h2>
                <p>{happy_count} happy moments</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # Current Emotion
            confidence_color = "#00C853" if self.current_confidence > 0.7 else "#FF9800"
            
            st.markdown(f"""
            <div class="analytics-card">
                <h3>🎯 Current Emotion</h3>
                <h2 style="color: {confidence_color}; margin: 0.5rem 0;">{self.current_emotion}</h2>
                <p>Confidence: {self.current_confidence:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            # Detection Stats
            st.markdown(f"""
            <div class="analytics-card">
                <h3>📈 Detection Stats</h3>
                <h2 style="color: #9C27B0; margin: 0.5rem 0;">{total_detections}</h2>
                <p>Total detections</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Emotion Distribution Chart
        st.subheader("📊 Emotion Distribution")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            try:
                import plotly.express as px
                
                # Prepare data for chart
                emotion_data = []
                for emotion, count in self.emotion_stats.items():
                    if count > 0:
                        emotion_data.append({
                            'Emotion': emotion,
                            'Count': count,
                            'Percentage': (count / total_detections) * 100
                        })
                
                if emotion_data:
                    df = pd.DataFrame(emotion_data)
                    fig = px.bar(df, x='Emotion', y='Count', 
                                color='Emotion',
                                color_discrete_map={
                                    'Angry': '#FF0000',
                                    'Disgust': '#008000',
                                    'Fear': '#800080',
                                    'Happy': '#FFFF00',
                                    'Sad': '#0000FF',
                                    'Surprise': '#FFA500',
                                    'Neutral': '#808080'
                                },
                                title='Emotion Distribution')
                    fig.update_layout(showlegend=False)
                    fig = self.apply_viz_theme(fig)
                    st.plotly_chart(fig, use_container_width=True, key=f"emotion_dist_{self.frame_count}")
            except ImportError:
                st.warning("Plotly not available for charts")
        
        with col2:
            # Emotion percentages
            st.write("**Emotion Percentages:**")
            for emotion in sorted(self.emotion_stats.keys()):
                count = self.emotion_stats[emotion]
                if count > 0:
                    percentage = (count / total_detections) * 100
                    color = self.get_emotion_color_hex(emotion)
                    st.markdown(f"<span style='color:{color};'>● {emotion}: {percentage:.1f}%</span>", 
                               unsafe_allow_html=True)
    
    def get_emotion_color_hex(self, emotion):
        """Get hex color for emotion"""
        colors = {
            'Angry': '#FF0000',
            'Disgust': '#008000', 
            'Fear': '#800080',
            'Happy': '#FFD700',
            'Sad': '#0000FF',
            'Surprise': '#FFA500',
            'Neutral': '#808080'
        }
        return colors.get(emotion, '#000000')
    
    def update_emotion_history(self, emotion, confidence):
        """Update emotion history for analytics"""
        # Convert numpy types to Python native types
        if hasattr(confidence, 'item'):
            confidence = confidence.item()  # Convert numpy scalar to Python float
        else:
            confidence = float(confidence)  # Convert to Python float
        
        self.current_emotion = emotion
        self.current_confidence = confidence
        
        # Add to history (limit to 100 entries)
        timestamp = datetime.now()
        self.emotion_history.append((emotion, confidence, timestamp))
        if len(self.emotion_history) > 100:
            self.emotion_history.pop(0)
        
        # Update user profile if logged in
        if st.session_state.user_logged_in:
            emotion_data = {
                'emotion': emotion,
                'confidence': confidence  # This is now a Python float
            }
            session_data = {
                'session_id': f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'start_time': datetime.now().isoformat()
            }
            self.profile_manager.update_user_profile(st.session_state.user_id, emotion_data, session_data)
    
    def render_webcam_mode(self):
        st.subheader("📷 Live Webcam Emotion Detection")
        
        # Debug info
        with st.expander("🔧 Debug Info"):
            if self.detector:
                st.write("✅ Emotion Detector initialized")
                if self.detector.model:
                    st.write("✅ Model loaded")
                else:
                    st.write("❌ Model NOT loaded")
                if self.detector.face_cascade and not self.detector.face_cascade.empty():
                    st.write("✅ Face detector loaded")
                else:
                    st.write("❌ Face detector NOT loaded")
            else:
                st.write("❌ Emotion Detector NOT initialized")
        
        # Initialize session state for webcam control
        if 'webcam_running' not in st.session_state:
            st.session_state.webcam_running = False
        
        st.warning("""
        **Note**: Webcam access requires permission from your browser. 
        Make sure you allow camera access when prompted.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if not st.session_state.webcam_running:
                if st.button('🎥 Start Webcam', type='primary', use_container_width=True):
                    st.session_state.webcam_running = True
                    # Reset analytics when starting new session
                    self.emotion_stats = {emotion: 0 for emotion in self.detector.emotion_labels.values()}
                    self.emotion_history = []
                    self.session_start_time = datetime.now()
                    self.frame_count = 0
                    st.rerun()
        
        with col2:
            if st.session_state.webcam_running:
                if st.button('🛑 Stop Webcam', type='secondary', use_container_width=True):
                    st.session_state.webcam_running = False
                    st.rerun()
        
        # Webcam feed
        if st.session_state.webcam_running:
            stframe = st.empty()
            status_text = st.empty()
            
            status_text.info("🔄 Initializing webcam...")
            
            # Initialize webcam
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            if not cap.isOpened():
                st.error("❌ Could not access webcam. Please check permissions.")
                st.session_state.webcam_running = False
                return
            
            status_text.success("✅ Webcam started! Press 'Stop Webcam' to end session.")
            
            # Show analytics alongside webcam
            analytics_placeholder = st.empty()
            
            # Process webcam frames
            while st.session_state.webcam_running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Failed to capture frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Detect faces and emotions every 3rd frame to improve performance
                if self.frame_count % 3 == 0 and self.detector:
                    faces, gray = self.detector.detect_faces(frame)
                    
                    if len(faces) > 0:
                        st.sidebar.success(f"👤 {len(faces)} face(s) detected")
                    
                    for (x, y, w, h) in faces:
                        face_roi = gray[y:y+h, x:x+w]
                        emotion, confidence, all_predictions = self.detector.predict_emotion(face_roi)
                        
                        # Update statistics
                        if emotion in self.emotion_stats:
                            self.emotion_stats[emotion] += 1
                        
                        # Update analytics
                        self.update_emotion_history(emotion, confidence)
                        
                        # Draw prediction on frame
                        frame = self.detector.draw_prediction(frame, x, y, w, h, emotion, confidence)
                
                # Convert BGR to RGB for Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, channels='RGB', use_container_width=True)
                
                # Update analytics display
                with analytics_placeholder.container():
                    self.render_real_time_analytics()
                
                self.frame_count += 1
                
                # Small delay to prevent freezing
                cv2.waitKey(1)
            
            # Cleanup when stopped
            cap.release()
            cv2.destroyAllWindows()
            status_text.info("📊 Webcam session ended.")
            
            # Show final analytics
            if sum(self.emotion_stats.values()) > 0:
                st.subheader("📋 Session Summary")
                self.show_emotion_statistics()
        else:
            st.info("👆 Press 'Start Webcam' to begin real-time emotion detection")
    
    def render_image_mode(self):
        st.subheader("🖼️ Image Emotion Detection")
        
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear face image for emotion detection"
        )
        
        if uploaded_file is not None:
            # Read and display image
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📤 Uploaded Image")
                st.image(image, use_container_width=True)
            
            with col2:
                st.subheader("🔍 Emotion Analysis")
                with st.spinner("Detecting emotions..."):
                    # Detect emotions
                    result_image, emotions_data = self.detect_emotion_in_image(image_np)
                    
                    if result_image is not None:
                        st.image(result_image, use_container_width=True,
                                caption="Emotion Detection Result")
                        
                        # Show emotion probabilities
                        self.show_emotion_probabilities(emotions_data)
                    else:
                        st.error("❌ No faces detected in the image. Please try another image.")
    
    def detect_emotion_in_image(self, image):
        """Detect emotions in uploaded image"""
        try:
            # Convert RGB to BGR for OpenCV
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image
            
            # Detect faces
            if self.detector:
                faces, gray = self.detector.detect_faces(image_bgr)
            else:
                return None, []
                
            emotions_data = []
            
            if len(faces) == 0:
                return None, []
            
            # Process each face
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                emotion, confidence, all_predictions = self.detector.predict_emotion(face_roi)
                
                # Convert confidence to Python float
                if hasattr(confidence, 'item'):
                    confidence = confidence.item()
                else:
                    confidence = float(confidence)
                
                # Convert probabilities to Python list
                if hasattr(all_predictions, 'tolist'):
                    all_predictions = all_predictions.tolist()
                
                # Draw prediction on image
                image_bgr = self.detector.draw_prediction(image_bgr, x, y, w, h, emotion, confidence)
                
                emotions_data.append({
                    'emotion': emotion,
                    'confidence': confidence,
                    'probabilities': all_predictions,
                    'bbox': (x, y, w, h)
                })
            
            # Convert back to RGB for display
            result_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            return result_image, emotions_data
            
        except Exception as e:
            st.error(f"❌ Error processing image: {e}")
            return None, []
    
    def show_emotion_probabilities(self, emotions_data):
        """Show emotion probabilities using Plotly"""
        if not emotions_data:
            return
        
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            
            for i, emotion_data in enumerate(emotions_data):
                st.subheader(f"👤 Face {i+1} Analysis")
                
                # Create probability chart
                if self.detector:
                    emotions = list(self.detector.emotion_labels.values())
                else:
                    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
                    
                probabilities = emotion_data['probabilities']
                
                fig = go.Figure(data=[
                    go.Bar(x=emotions, y=probabilities,
                          marker_color=['#FF0000', '#008000', '#800080', '#FFFF00', 
                                      '#0000FF', '#FFA500', '#808080'])
                ])
                
                fig.update_layout(
                    title=f"Emotion Probabilities (Detected: {emotion_data['emotion']})",
                    xaxis_title="Emotions",
                    yaxis_title="Probability",
                    yaxis_range=[0, 1],
                    showlegend=False
                )
                
                fig = self.apply_viz_theme(fig)
                st.plotly_chart(fig, use_container_width=True, key=f"face_prob_{i}")
                
                # Show confidence
                st.metric(
                    label=f"Detected Emotion: {emotion_data['emotion']}",
                    value=f"{emotion_data['confidence']:.2%}",
                    delta="High Confidence" if emotion_data['confidence'] > 0.7 else "Medium Confidence"
                )
        except ImportError:
            st.warning("Plotly not installed. Install with: pip install plotly")
    
    def show_emotion_statistics(self):
        """Show emotion detection statistics"""
        st.subheader("📈 Detection Statistics")
        
        if sum(self.emotion_stats.values()) == 0:
            st.info("No emotions detected during the session")
            return
        
        try:
            import plotly.express as px
            import pandas as pd
            
            # Create pie chart
            fig = px.pie(
                values=list(self.emotion_stats.values()),
                names=list(self.emotion_stats.keys()),
                title="Emotion Distribution",
                color=list(self.emotion_stats.keys()),
                color_discrete_map={
                    'Angry': '#FF0000',
                    'Disgust': '#008000',
                    'Fear': '#800080',
                    'Happy': '#FFFF00',
                    'Sad': '#0000FF',
                    'Surprise': '#FFA500',
                    'Neutral': '#808080'
                }
            )
            
            fig = self.apply_viz_theme(fig)
            st.plotly_chart(fig, use_container_width=True, key="final_stats_pie")
            
            # Show statistics table
            stats_df = pd.DataFrame({
                'Emotion': list(self.emotion_stats.keys()),
                'Count': list(self.emotion_stats.values()),
                'Percentage': [f"{(count/sum(self.emotion_stats.values()))*100:.1f}%" 
                              for count in self.emotion_stats.values()]
            }).sort_values('Count', ascending=False)
            
            st.dataframe(stats_df, use_container_width=True)
            
        except ImportError:
            # Simple text display if plotly not available
            st.write("Emotion Statistics:")
            for emotion, count in sorted(self.emotion_stats.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / sum(self.emotion_stats.values())) * 100
                st.write(f"- {emotion}: {count} ({percentage:.1f}%)")

    def render_user_profile_dashboard(self):
        """Render user profile dashboard with personal insights"""
        if not st.session_state.user_logged_in:
            st.info("🔐 Please login to view your personal emotion profile")
            return
        
        st.subheader("👤 Personal Emotion Profile")
        
        user_stats = st.session_state.user_stats
        if not user_stats:
            st.info("No emotion data recorded yet. Start a session to build your profile!")
            return
        
        # Personal insights cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Sessions", user_stats['total_sessions'])
        with col2:
            st.metric("Emotions Tracked", user_stats['total_emotions'])
        with col3:
            st.metric("Dominant Emotion", user_stats['dominant_emotion'])
        with col4:
            st.metric("Avg Confidence", f"{user_stats['avg_confidence']:.1%}")
        
        # Personal emotion distribution
        st.subheader("📊 Your Emotion Patterns")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            try:
                import plotly.express as px
                
                # Create personalized emotion distribution
                emotion_dist = user_stats['emotion_distribution']
                df_personal = pd.DataFrame({
                    'Emotion': list(emotion_dist.keys()),
                    'Count': list(emotion_dist.values())
                })
                
                fig_personal = px.pie(
                    df_personal, values='Count', names='Emotion',
                    title='Your Personal Emotion Distribution',
                    color='Emotion',
                    color_discrete_map={
                        'Angry': '#FF0000', 'Disgust': '#008000', 'Fear': '#800080',
                        'Happy': '#FFFF00', 'Sad': '#0000FF', 'Surprise': '#FFA500', 'Neutral': '#808080'
                    }
                )
                
                fig_personal = self.apply_viz_theme(fig_personal)
                st.plotly_chart(fig_personal, use_container_width=True, key="personal_dist")
                
            except ImportError:
                st.warning("Plotly not available for charts")
        
        with col2:
            # Personal insights
            st.write("**Your Insights:**")
            st.write(f"🏆 Most frequent: **{user_stats['dominant_emotion']}**")
            st.write(f"😊 Happiness ratio: **{user_stats['happiness_ratio']:.1f}%**")
            st.write(f"📅 Recent activity: **{user_stats['recent_activity']}** emotions this week")
            
            # Progress indicators
            if user_stats['total_emotions'] > 50:
                st.success("🎯 Great consistency! Keep tracking your emotions.")
            elif user_stats['total_emotions'] > 20:
                st.info("📈 Good progress! You're building valuable data.")
            else:
                st.warning("🔍 Keep going! More data will provide better insights.")

    def render_advanced_visualizations(self):
        """Render advanced visualization options"""
        st.subheader("🎨 Advanced Emotion Visualizations")
        
        if not self.emotion_history and (not st.session_state.user_logged_in or not st.session_state.user_stats):
            st.info("🔍 Start a webcam session or upload images to see advanced visualizations")
            return
        
        # Use user data if available, otherwise use session data
        if st.session_state.user_logged_in and st.session_state.user_stats:
            emotion_data = st.session_state.user_stats['emotion_distribution']
            data_title = "Your Emotion Data"
        else:
            emotion_data = self.emotion_stats
            data_title = "Current Session Data"
        
        st.write(f"**Displaying:** {data_title}")
        
        # Visualization type selection
        viz_type = st.selectbox(
            "Choose Visualization Type",
            ["Emotion Radar", "Confidence Timeline", "3D Landscape", "Comparative Analysis"],
            index=0
        )
        
        if viz_type == "Emotion Radar":
            self.render_emotion_radar(emotion_data)
        elif viz_type == "Confidence Timeline":
            self.render_confidence_timeline()
        elif viz_type == "3D Landscape":
            self.render_3d_landscape(emotion_data)
        elif viz_type == "Comparative Analysis":
            self.render_comparative_analysis(emotion_data)

    def render_emotion_radar(self, emotion_data):
        """Render emotion radar chart"""
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            
            emotions = list(emotion_data.keys())
            values = list(emotion_data.values())
            
            # Normalize values for radar chart
            max_val = max(values) if values else 1
            normalized_values = [v / max_val * 100 for v in values]
            
            fig = go.Figure(data=
                go.Scatterpolar(
                    r=normalized_values + [normalized_values[0]],  # Close the circle
                    theta=emotions + [emotions[0]],  # Close the circle
                    fill='toself',
                    line=dict(color='#1f77b4'),
                    fillcolor='rgba(31, 119, 180, 0.3)'
                )
            )
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                showlegend=False,
                title="Emotion Radar Chart"
            )
            
            st.plotly_chart(fig, use_container_width=True, key="emotion_radar")
            
        except Exception as e:
            st.error(f"Error creating radar chart: {e}")

    def render_confidence_timeline(self):
        """Render confidence timeline visualization"""
        try:
            import plotly.express as px
            
            if not self.emotion_history:
                st.info("No emotion history available for timeline")
                return
            
            timeline_data = []
            for i, (emotion, confidence, timestamp) in enumerate(self.emotion_history):
                timeline_data.append({
                    'Time': i,
                    'Emotion': emotion,
                    'Confidence': confidence,
                    'Timestamp': timestamp
                })
            
            df_timeline = pd.DataFrame(timeline_data)
            
            fig = px.line(df_timeline, x='Time', y='Confidence', color='Emotion',
                         title='Emotion Confidence Timeline',
                         hover_data=['Timestamp'])
            
            fig = self.apply_viz_theme(fig)
            st.plotly_chart(fig, use_container_width=True, key="confidence_timeline")
            
        except Exception as e:
            st.error(f"Error creating timeline: {e}")

    def render_3d_landscape(self, emotion_data):
        """Render 3D emotion landscape"""
        try:
            import plotly.graph_objects as go
            
            emotions = list(emotion_data.keys())
            values = list(emotion_data.values())
            
            # Create 3D surface data
            x = np.arange(len(emotions))
            y = np.arange(len(emotions))
            X, Y = np.meshgrid(x, y)
            Z = np.zeros((len(emotions), len(emotions)))
            
            # Create emotion intensity landscape
            for i in range(len(emotions)):
                for j in range(len(emotions)):
                    Z[i,j] = values[i] * values[j] / 100  # Normalize
            
            fig = go.Figure(data=[
                go.Surface(
                    x=X, y=Y, z=Z,
                    colorscale='Viridis',
                    opacity=0.8
                )
            ])
            
            fig.update_layout(
                title='3D Emotion Intensity Landscape',
                scene=dict(
                    xaxis=dict(title='Emotions', tickvals=x, ticktext=emotions),
                    yaxis=dict(title='Emotions', tickvals=y, ticktext=emotions),
                    zaxis=dict(title='Intensity'),
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
                ),
                width=800,
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True, key="3d_landscape")
            
        except Exception as e:
            st.error(f"Error creating 3D landscape: {e}")

    def render_comparative_analysis(self, emotion_data):
        """Render comparative analysis visualization"""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            emotions = list(emotion_data.keys())
            values = list(emotion_data.values())
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Bar Chart', 'Pie Chart', 'Line Chart', 'Area Chart'),
                specs=[[{"type": "bar"}, {"type": "pie"}],
                       [{"type": "scatter"}, {"type": "scatter"}]]
            )
            
            # Bar chart
            fig.add_trace(
                go.Bar(x=emotions, y=values, name="Count"),
                row=1, col=1
            )
            
            # Pie chart
            fig.add_trace(
                go.Pie(labels=emotions, values=values, name="Distribution"),
                row=1, col=2
            )
            
            # Line chart
            fig.add_trace(
                go.Scatter(x=emotions, y=values, mode='lines+markers', name="Trend"),
                row=2, col=1
            )
            
            # Area chart
            fig.add_trace(
                go.Scatter(x=emotions, y=values, fill='tozeroy', name="Area"),
                row=2, col=2
            )
            
            fig.update_layout(
                height=600,
                title_text="Comparative Emotion Analysis",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True, key="comparative_analysis")
            
        except Exception as e:
            st.error(f"Error creating comparative analysis: {e}")

    def run(self):
        # Ensure session state is initialized
        self.initialize_session_state()
        
        self.render_sidebar()
        self.render_visualization_controls()
        
        # Navigation
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎮 Navigation")
        
        app_mode = st.sidebar.selectbox(
            "Choose Mode",
            ["🏠 Home", "👤 My Profile", "📷 Live Webcam", "🖼️ Image Upload", "🎨 Visualizations"],
            index=0
        )
        
        # Render selected mode
        if app_mode == "🏠 Home":
            self.render_home()
        elif app_mode == "👤 My Profile":
            self.render_user_profile_dashboard()
        elif app_mode == "📷 Live Webcam":
            self.render_webcam_mode()
        elif app_mode == "🖼️ Image Upload":
            self.render_image_mode()
        elif app_mode == "🎨 Visualizations":
            self.render_advanced_visualizations()

# Run the app
if __name__ == "__main__":
    app = EmotionDetectionApp()
    app.run()