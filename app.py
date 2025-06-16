import streamlit as st
import numpy as np
import cv2
import os
import torch
from datetime import datetime
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import pandas as pd
import base64
import requests
import gspread
from google.oauth2.service_account import Credentials

# Page Config 
st.set_page_config(page_title="Face Attendance", layout="centered")

# 🖼️ Set background image
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def set_background(image_file):
    encoded = get_base64_image(image_file)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
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

# 🔐 Google Sheets Setup
SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
service_account_info = st.secrets["gcp_service_account"]
creds = Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
gc = gspread.authorize(creds)
SHEET_ID = '1lO0qt1EWZAwXjhRUOk19igYwI2rNyx5hLkG4wLyUkzc'

# 📍 Constants
INDIANA_LOCATION = (12.8697, 74.8426)
LOCATION_RADIUS_KM = 0.5
os.makedirs("data", exist_ok=True)

# 🧠 Face Recognition Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(image_size=160, margin=20, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# 📦 Session State Init
if "embeddings" not in st.session_state:
    st.session_state.embeddings = {}
if "attendance" not in st.session_state:
    st.session_state.attendance = []

# 🔧 Utility Functions
def extract_face(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_tensor = mtcnn(Image.fromarray(img_rgb))
    return face_tensor.unsqueeze(0).to(device) if face_tensor is not None else None

def get_embedding(face_tensor):
    with torch.no_grad():
        return model(face_tensor)[0].cpu().numpy()

def is_match(known, candidate, thresh=0.9):
    return np.linalg.norm(known - candidate) < thresh

def get_user_location():
    try:
        loc = requests.get("https://ipinfo.io/json").json()['loc'].split(',')
        return float(loc[0]), float(loc[1])
    except:
        return None

def haversine(loc1, loc2):
    from math import radians, sin, cos, sqrt, atan2
    R = 6371
    lat1, lon1 = map(radians, loc1)
    lat2, lon2 = map(radians, loc2)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def is_within_location(user_loc):
    return haversine(user_loc, INDIANA_LOCATION) <= LOCATION_RADIUS_KM if user_loc else False

def append_attendance(name, date, time):
    try:
        worksheet = gc.open_by_key(SHEET_ID).worksheet(date)
    except gspread.exceptions.WorksheetNotFound:
        worksheet = gc.open_by_key(SHEET_ID).add_worksheet(title=date, rows="1000", cols="3")
        worksheet.append_row(["Name", "Date", "Time"])
    worksheet.append_row([name, date, time])
    return True

def get_today_attendance():
    date = datetime.now().strftime("%Y-%m-%d")
    try:
        worksheet = gc.open_by_key(SHEET_ID).worksheet(date)
        return worksheet.get_all_records()
    except:
        return []

# 🌐 UI
st.title("📸 Face Recognition Attendance")

menu = st.sidebar.selectbox("Menu", [
    "Register Face",
    "Take Attendance",
    "View Attendance Sheet",
    "View Registered Users"
])
admin_password = st.sidebar.text_input("🔐 Admin Password", type="password")

# 👤 Register Face
if menu == "Register Face":
    st.subheader("📝 Register New Face")
    name = st.text_input("Enter your name")
    uploaded = st.file_uploader("Upload a picture", type=["jpg", "jpeg", "png"])

    if uploaded and name:
        img = np.array(Image.open(uploaded))
        face_tensor = extract_face(img)
        if face_tensor is not None:
            emb = get_embedding(face_tensor)
            st.session_state.embeddings[name] = emb
            np.savez("data/registered_faces.npz", **st.session_state.embeddings)
            st.success(f"✅ Registered {name}")
        else:
            st.error("❌ No face detected.")

# 📷 Take Attendance
elif menu == "Take Attendance":
    st.subheader("📷 Take Attendance")
    captured = st.camera_input("Take your photo")

    if captured:
        user_loc = get_user_location()
        if not is_within_location(user_loc):
            st.error("🚫 You are not in Indiana Hospital.")
        else:
            file_bytes = np.asarray(bytearray(captured.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            face_tensor = extract_face(img)

            if face_tensor is not None:
                emb = get_embedding(face_tensor)

                if not st.session_state.embeddings and os.path.exists("data/registered_faces.npz"):
                    data = np.load("data/registered_faces.npz")
                    st.session_state.embeddings = {n: data[n] for n in data.files}

                for name, known_emb in st.session_state.embeddings.items():
                    if is_match(known_emb, emb):
                        now = datetime.now()
                        date = now.strftime("%Y-%m-%d")
                        time = now.strftime("%H:%M:%S")
                        record = {"Name": name, "Date": date, "Time": time}
                        if record not in st.session_state.attendance:
                            st.session_state.attendance.append(record)
                            if append_attendance(name, date, time):
                                st.success(f"✅ Attendance marked for {name}")
                        else:
                            st.info("ℹ️ Already marked today.")
                        break
                else:
                    st.warning("⚠️ Face not recognized.")
            else:
                st.error("❌ No face detected.")

# 📅 View Today’s Attendance
elif menu == "View Attendance Sheet":
    st.subheader("📅 Today's Attendance")
    today_records = get_today_attendance()
    if today_records:
        df = pd.DataFrame(today_records)
        st.dataframe(df)
    else:
        st.info("📭 No attendance found for today.")

# 📂 View Registered Users
elif menu == "View Registered Users":
    st.subheader("👥 Registered Users")
    if os.path.exists("data/registered_faces.npz"):
        with np.load("data/registered_faces.npz") as data:
            names = list(data.files)
        if names:
            st.markdown("### Registered:")
            for name in names:
                st.markdown(f"- {name}")
            if admin_password == "secret123":
                if st.button("❌ Clear Registered Users"):
                    os.remove("data/registered_faces.npz")
                    st.session_state.embeddings = {}
                    st.success("✅ Cleared all users.")
            else:
                st.warning("🔒 Enter correct admin password to clear users.")
        else:
            st.info("📭 No users found.")
    else:
        st.info("📭 No registered users yet.")
