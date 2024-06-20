import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import numpy as np
import mediapipe as mp
import google.generativeai as genai
from PIL import Image

# Configuration for Google Generative AI
genai.configure(api_key="AIzaSyDiyJCpfu8wzNXcVbHhPAqlBZ0hgFpsCKY")
model = genai.GenerativeModel('gemini-1.5-flash')

class HandDetector:
    def __init__(self, detection_confidence=0.5, tracking_confidence=0.5):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

    def find_hands(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        return results

    def draw_hands(self, img, results):
        mp_drawing = mp.solutions.drawing_utils
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        return img

    def fingers_up(self, results):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                fingers = []
                for id in [4, 8, 12, 16, 20]:
                    if id == 4:
                        if hand_landmarks.landmark[id].x > hand_landmarks.landmark[id - 1].x:
                            fingers.append(1)
                        else:
                            fingers.append(0)
                    else:
                        if hand_landmarks.landmark[id].y < hand_landmarks.landmark[id - 2].y:
                            fingers.append(1)
                        else:
                            fingers.append(0)
                return fingers
        return []

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.hand_detector = HandDetector()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = self.hand_detector.find_hands(img)
        img = self.hand_detector.draw_hands(img, results)
        fingers = self.hand_detector.fingers_up(results)
        
        # Check for specific finger gestures
        if fingers == [1, 1, 1, 1, 1]:  # All fingers up
            img = cv2.putText(img, "All Fingers Up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("Hand Gesture Recognition")
webrtc_ctx = webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, video_processor_factory=VideoProcessor, media_stream_constraints={"video": True, "audio": False})

