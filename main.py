import cv2
import numpy as np
import mediapipe as mp
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import streamlit as st
from PIL import Image
import google.generativeai as genai

# Configuration for Google Generative AI
genai.configure(api_key="AIzaSyDiyJCpfu8wzNXcVbHhPAqlBZ0hgFpsCKY")
model = genai.GenerativeModel('gemini-1.5-flash')

# Hand Detection class using MediaPipe
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

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.detector = HandDetector()
        self.canvas = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        if self.canvas is None:
            self.canvas = np.zeros_like(img)

        results = self.detector.find_hands(img)
        img = self.detector.draw_hands(img, results)
        fingers = self.detector.fingers_up(results)

        # Process the fingers to draw on the canvas
        if fingers == [0, 1, 0, 0, 0]:
            current_pos = (int(hand_landmarks.landmark[8].x * img.shape[1]), int(hand_landmarks.landmark[8].y * img.shape[0]))
            if self.prev_pos is None:
                self.prev_pos = current_pos
            cv2.line(self.canvas, self.prev_pos, current_pos, (255, 0, 255), 10)
            self.prev_pos = current_pos
        elif fingers == [1, 0, 0, 0, 0]:
            self.canvas = np.zeros_like(img)

        img = cv2.addWeighted(img, 0.7, self.canvas, 0.3, 0)

        # Generate AI response if needed
        if fingers == [1, 1, 1, 1, 0]:
            pil_image = Image.fromarray(self.canvas)
            response = model.generate_content(["Solve this math problem", pil_image])
            st.write(response.text)

        return img

st.title("Webcam Hand Tracking and AI Interaction")
st.write("This application uses your webcam to detect and track hands in real-time using MediaPipe and OpenCV, and interacts with Google Generative AI.")

webrtc_streamer(key="unique_key", video_transformer_factory=VideoTransformer)

st.write("Note: Ensure you allow the browser to access your webcam.")
