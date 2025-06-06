import streamlit as st
import numpy as np
import cv2
import os
import torch
import random
from datetime import datetime
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import pandas as pd
import base64
import requests
from geopy.distance import geodesic
import gspread
from google.oauth2.service_account import Credentials

# Define scope
SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']

# Load credentials from Streamlit Secrets
service_account_info = st.secrets["gcp_service_account"]

# Create credentials object
creds = Credentials.from_service_account_info(service_account_info, scopes=SCOPES)

# Authorize gspread client
gc = gspread.authorize(creds)

# Open the Google Sheet
SHEET_ID = '1lO0qt1EWZAwXjhRUOk19igYwI2rNyx5hLkG4wLyUkzc'
sheet = gc.open_by_key(SHEET_ID).sheet1

INDIANA_LOCATION = (12.8697, 74.8426)  # Latitude, Longitude
LOCATION_RADIUS_KM = 0.5  # Acceptable distance in kilometers

def get_user_location():
    try:
        response = requests.get("https://ipinfo.io/json")
        data = response.json()
        loc = data['loc'].split(',')
        return float(loc[0]), float(loc[1])
    except:
        return None

def append_attendance(name, date, time):
    # Append a new row: [Name, Date, Time]
    try:
        sheet.append_row([name, date, time])
        return True
    except Exception as e:
        st.error(f"Failed to write to Google Sheet: {e}")
        return False

from datetime import datetime
import pytz

# Define IST timezone
ist = pytz.timezone('Asia/Kolkata')

def get_today_attendance():
    try:
        all_records = sheet.get_all_records()
        today_str = datetime.now(ist).strftime("%Y-%m-%d")

        # Debug: Show raw data temporarily
        #st.write("All Records:", all_records)

        # Ensure 'Date' column matches today's date
        today_records = [rec for rec in all_records if rec.get('Date') == today_str]

        return today_records
    except Exception as e:
        st.error(f"❌ Failed to read Google Sheet: {e}")
        return []

# Page config
st.set_page_config(page_title="Face Attendance", layout="centered")

# Set background
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return encoded

def set_background(image_file):
    encoded_image = get_base64_image(image_file)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("background.jpg")

from streamlit_autorefresh import st_autorefresh
from zoneinfo import ZoneInfo

# Refresh clock every 60 seconds
st_autorefresh(interval=60000, limit=None, key="clock_refresh")

# Set IST timezone
ist = ZoneInfo("Asia/Kolkata")
current_time = datetime.now(ist).strftime("%Y-%m-%d %H:%M")

# Display time in sidebar
st.sidebar.markdown(f"🕒 **Current Time (IST):** `{current_time}`")

# Unified Select Action
menu = st.sidebar.selectbox(
    "Select Action",
    [
        "Register Face",
        "Take Attendance",
        "View Attendance Sheet",
        "View Registered Users"
    ],
    key="action_selectbox"
)

# Admin password input
admin_password = st.sidebar.text_input("🔐 Admin Password", type="password")

# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(image_size=160, margin=20, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Session state
if "embeddings" not in st.session_state:
    st.session_state.embeddings = {}
if "attendance" not in st.session_state:
    st.session_state.attendance = []

# Directory
os.makedirs("data", exist_ok=True)

# Utilities
def extract_face(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_tensor = mtcnn(Image.fromarray(img_rgb))
    if face_tensor is not None:
        return face_tensor.unsqueeze(0).to(device)
    return None

def get_embedding(face_tensor):
    with torch.no_grad():
        embedding = model(face_tensor)
    return embedding[0].cpu().numpy()

def is_match(known, candidate, thresh=0.9):
    return np.linalg.norm(known - candidate) < thresh

# Main App Title
st.markdown("""
<div style="
    background-color: #ecf6f7; 
    padding: 1.5rem; 
    border-radius: 15px; 
    text-align: center; 
    box-shadow: 0 4px 10px rgba(43, 103, 119, 0.15);
">
    <h2 style="color: #2b6777; margin-bottom: 0.5rem;"> Presencia - A Face Attendance System</h2>
    <p style="font-size: 20px; color: #2b6777; font-style: italic;">Look once. You're marked present.</p>
</div>
""", unsafe_allow_html=True)

if menu == "Register Face":
    st.markdown('<h3 style="text-align: center; color: #2b6777;"> Register New Face</h3>', unsafe_allow_html=True)
    name = st.text_input("Enter your name")
    uploaded_picture = st.file_uploader("Upload a picture", type=["jpg", "jpeg", "png"])

    if uploaded_picture and name:
        # Load saved embeddings if not already in session
        if os.path.exists("data/registered_faces.npz"):
            data = np.load("data/registered_faces.npz")
            for key in data.files:
                if key not in st.session_state.embeddings:
                    st.session_state.embeddings[key] = data[key]
        image = Image.open(uploaded_picture)
        img = np.array(image)
        face_tensor = extract_face(img)
        if face_tensor is not None:
            emb = get_embedding(face_tensor)
            st.session_state.embeddings[name] = emb
            np.savez("data/registered_faces.npz", **st.session_state.embeddings)
            st.success(f"✅ Face registered for {name}")
        else:
            st.error("❌ No face detected. Try a clearer image.")

elif menu == "Take Attendance":
    st.subheader("Take Attendance")

    import mediapipe as mp
    import time

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    MOUTH = [78, 308, 13, 14, 312, 82]

    def euclidean(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def eye_aspect_ratio(landmarks, eye_points):
        p1 = landmarks[eye_points[0]]
        p2 = landmarks[eye_points[1]]
        p3 = landmarks[eye_points[2]]
        p4 = landmarks[eye_points[3]]
        p5 = landmarks[eye_points[4]]
        p6 = landmarks[eye_points[5]]
        vertical1 = euclidean(p2, p6)
        vertical2 = euclidean(p3, p5)
        horizontal = euclidean(p1, p4)
        return (vertical1 + vertical2) / (2.0 * horizontal)

    def mouth_aspect_ratio(landmarks):
        top_lip = landmarks[13]
        bottom_lip = landmarks[14]
        left_corner = landmarks[78]
        right_corner = landmarks[308]
        vertical = euclidean(top_lip, bottom_lip)
        horizontal = euclidean(left_corner, right_corner)
        return vertical / horizontal

    def head_yaw(landmarks, frame_width):
        nose_tip = landmarks[1]
        left_cheek = landmarks[234]
        right_cheek = landmarks[454]
        center_x = (left_cheek[0] + right_cheek[0]) / 2
        diff = nose_tip[0] - center_x
        norm_diff = diff / frame_width
        return norm_diff

    EAR_THRESH = 0.23
    EAR_CONSEC_FRAMES = 3
    MOUTH_OPEN_THRESH = 0.6
    HEAD_TURN_THRESH = 0.08


    if "attendance_started" not in st.session_state:
        st.session_state.attendance_started = False
    if "challenges" not in st.session_state:
        st.session_state.challenges = []
    if "current_task_index" not in st.session_state:
        st.session_state.current_task_index = 0
    if "task_0_done" not in st.session_state:
        st.session_state.task_0_done = False
    if "task_1_done" not in st.session_state:
        st.session_state.task_1_done = False

    if not st.session_state.attendance_started:
        location = get_user_location()
        if location is None:
            st.error("Unable to determine your location. Please check your internet.")
        elif geodesic(location, INDIANA_LOCATION).km > LOCATION_RADIUS_KM:
            st.warning("You must be at Indiana Hospital & Heart Institute, Mangalore to mark attendance.")
        else:
            if st.button("📸 Start Attendance"):
                pool = ["blink", "turn_head", "open_mouth"]
                st.session_state.challenges = random.sample(pool, 2)
                st.session_state.current_task_index = 0
                st.session_state.task_0_done = False
                st.session_state.task_1_done = False
                st.session_state.attendance_started = True
                st.rerun()

        if st.button("📸 Start Attendance"):
            pool = ["blink", "turn_head", "open_mouth"]
            st.session_state.challenges = random.sample(pool, 2)
            st.session_state.current_task_index = 0
            st.session_state.task_0_done = False
            st.session_state.task_1_done = False
            st.session_state.attendance_started = True
            st.rerun()
        else:
            st.info("Click the '📸 Start Attendance' button to begin.")

    else:
        stframe = st.empty()
        message_placeholder = st.empty()

        cap = cv2.VideoCapture(0)

        blink_counter = 0
        blink_frame_counter = 0

        start_time = time.time()

        # Guard: if index is out of range, all tasks are done
        if st.session_state.current_task_index < len(st.session_state.challenges):
            current_task = st.session_state.challenges[st.session_state.current_task_index]
            st.markdown(f" Please perform this task: **{current_task.replace('_', ' ').title()}**")
        else:
            # All tasks done, mark attendance and reset
            cap.release()
            stframe.empty()
            message_placeholder.info("⏳ Marking your attendance... Please wait.")

            ret2, frame2 = cv2.VideoCapture(0).read()
            cv2.VideoCapture(0).release()

            if ret2:
                face_tensor = extract_face(frame2)
                matched_name = None

                if face_tensor is not None:
                    emb = get_embedding(face_tensor)

                    if not st.session_state.embeddings and os.path.exists("data/registered_faces.npz"):
                        data = np.load("data/registered_faces.npz")
                        st.session_state.embeddings = {name: data[name] for name in data.files}

                    for name, db_emb in st.session_state.embeddings.items():
                        if is_match(db_emb, emb):
                            matched_name = name
                            break

                if matched_name:
                    now = datetime.now(ist)
                    # Check if user already marked attendance today (optional: you can skip or implement this check)
                    today_records = get_today_attendance()
                    already_marked = any(rec["Name"] == matched_name for rec in today_records)

                    if not already_marked:
                        success = append_attendance(matched_name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"))
                        if success:
                            message_placeholder.success(f" Your attendance is marked, {matched_name}.")
                        else:
                            message_placeholder.error("Failed to mark attendance in Google Sheets.")
                    else:
                        message_placeholder.info(f" {matched_name}, you have already marked attendance today.")
                else:
                    message_placeholder.warning("⚠️ Face not recognized. Please try again.")
            else:
                message_placeholder.error(" Failed to capture face for attendance.")

            # Reset session states
            st.session_state.attendance_started = False
            st.session_state.challenges = []
            st.session_state.current_task_index = 0
            st.session_state.task_0_done = False
            st.session_state.task_1_done = False

            st.rerun()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error(" Failed to capture video.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            task_done = False

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in face_landmarks.landmark]

                if current_task == "blink":
                    left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
                    right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
                    avg_ear = (left_ear + right_ear) / 2.0

                    if avg_ear < EAR_THRESH:
                        blink_frame_counter += 1
                    else:
                        if blink_frame_counter >= EAR_CONSEC_FRAMES:
                            blink_counter += 1
                        blink_frame_counter = 0

                    cv2.putText(frame, f"Blinks: {blink_counter}", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    if blink_counter >= 1:
                        task_done = True

                elif current_task == "open_mouth":
                    mar = mouth_aspect_ratio(landmarks)
                    cv2.putText(frame, f"Mouth Aspect Ratio: {mar:.2f}", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if mar > MOUTH_OPEN_THRESH:
                        task_done = True


                elif current_task == "turn_head":

                    yaw = head_yaw(landmarks, frame.shape[1])

                    cv2.putText(frame, f"Head Yaw: {yaw:.3f}", (30, 50),

                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    if abs(yaw) > HEAD_TURN_THRESH:  # ✅ Corrected logic

                        task_done = True

            else:
                cv2.putText(frame, "No face detected", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            stframe.image(frame, channels="BGR")

            # Timeout per task = 20 seconds
            if time.time() - start_time > 15:
                cap.release()
                stframe.empty()
                message_placeholder.warning(" Time out. You didn't complete the task.")
                st.session_state.attendance_started = False
                st.session_state.challenges = []
                st.session_state.current_task_index = 0
                st.session_state.task_0_done = False
                st.session_state.task_1_done = False
                break

            if task_done:
                st.success(
                    f"✅ Task {st.session_state.current_task_index + 1} completed: **{current_task.replace('_', ' ').title()}**")

                if st.session_state.current_task_index == 0:
                    st.session_state.task_0_done = True
                else:
                    st.session_state.task_1_done = True

                st.session_state.current_task_index += 1
                cap.release()
                stframe.empty()
                message_placeholder.empty()
                time.sleep(4)
                st.rerun()

elif menu == "View Attendance Sheet":
    st.subheader("📅 Today's Attendance")

    today_records = get_today_attendance()
    if today_records:
        df = pd.DataFrame(today_records)
        st.dataframe(df)
    else:
        st.info("No attendance marked today.")


# View Registered Students Page
elif menu == "View Registered Users":
    st.subheader("👥 Registered Users")
    if os.path.exists("data/registered_faces.npz"):
        with np.load("data/registered_faces.npz") as data:
            registered_names = list(data.files)
        if registered_names:
            for name in registered_names:
                st.markdown(f"- {name}")
            if admin_password == "secret123":
                if st.button("❌ Clear Registered Users"):
                    os.remove("data/registered_faces.npz")
                    st.session_state.embeddings = {}
                    st.success("✅ Registered Users cleared.")
            else:
                st.warning("🔒 Enter correct admin password to clear registered Users.")
        else:
            st.info("No Users found in the data file.")
    else:
        st.info("📭 No registered Users found.")
