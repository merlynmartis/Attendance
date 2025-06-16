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

# Google Sheets Setup
SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
service_account_info = st.secrets["gcp_service_account"]
creds = Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
gc = gspread.authorize(creds)
SHEET_ID = 'YOUR_SHEET_ID'  # Replace with your sheet ID

# Constants
INDIANA_LOCATION = (12.8697, 74.8426)
LOCATION_RADIUS_KM = 0.5

# Streamlit UI Setup
st.set_page_config(page_title="Face Attendance", layout="centered")

# Model Initialization
device = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(image_size=160, margin=20, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Session State
if "embeddings" not in st.session_state:
    st.session_state.embeddings = {}
if "attendance" not in st.session_state:
    st.session_state.attendance = []

os.makedirs("data", exist_ok=True)

# Utility Functions
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

def get_user_location():
    try:
        response = requests.get("https://ipinfo.io/json")
        loc = response.json()["loc"].split(',')
        return float(loc[0]), float(loc[1])
    except:
        return None

def haversine(loc1, loc2):
    from math import radians, sin, cos, sqrt, atan2
    R = 6371
    lat1, lon1 = map(radians, loc1)
    lat2, lon2 = map(radians, loc2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def is_within_location(user_loc):
    if user_loc is None:
        return False
    return haversine(user_loc, INDIANA_LOCATION) <= LOCATION_RADIUS_KM

def append_attendance(name, date, time):
    try:
        try:
            worksheet = gc.open_by_key(SHEET_ID).worksheet(date)
        except gspread.exceptions.WorksheetNotFound:
            worksheet = gc.open_by_key(SHEET_ID).add_worksheet(title=date, rows="1000", cols="3")
            worksheet.append_row(["Name", "Date", "Time"])
        worksheet.append_row([name, date, time])
        return True
    except Exception as e:
        st.error(f"❌ Failed to write to Google Sheet: {e}")
        return False

# UI Logic
menu = st.sidebar.selectbox("Select Action", ["Register Face", "Take Attendance"])
admin_password = st.sidebar.text_input("🔐 Admin Password", type="password")

if menu == "Register Face":
    st.header("📝 Register New Face")
    name = st.text_input("Enter your name")
    uploaded = st.file_uploader("Upload a picture", type=["jpg", "jpeg", "png"])
    
    if uploaded and name:
        image = Image.open(uploaded)
        img = np.array(image)
        face_tensor = extract_face(img)

        if face_tensor is not None:
            emb = get_embedding(face_tensor)
            st.session_state.embeddings[name] = emb
            np.savez("data/registered_faces.npz", **st.session_state.embeddings)
            st.success(f"✅ Registered face for {name}")
        else:
            st.error("❌ No face detected.")

elif menu == "Take Attendance":
    st.header("📷 Take Attendance")
    uploaded = st.camera_input("Capture your face")

    if uploaded:
        user_loc = get_user_location()
        if not is_within_location(user_loc):
            st.error("📍 You are not within Indiana Hospital. Attendance denied.")
        else:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            face_tensor = extract_face(img)

            if face_tensor is not None:
                emb = get_embedding(face_tensor)

                if not st.session_state.embeddings and os.path.exists("data/registered_faces.npz"):
                    data = np.load("data/registered_faces.npz")
                    st.session_state.embeddings = {name: data[name] for name in data.files}

                for name, db_emb in st.session_state.embeddings.items():
                    if is_match(db_emb, emb):
                        now = datetime.now()
                        date = now.strftime("%Y-%m-%d")
                        time = now.strftime("%H:%M:%S")
                        record = {"Name": name, "Date": date, "Time": time}

                        if record not in st.session_state.attendance:
                            st.session_state.attendance.append(record)
                            uploaded = append_attendance(name, date, time)
                            if uploaded:
                                st.success(f"✅ Attendance marked for {name}")
                        else:
                            st.info(f"ℹ️ Attendance already marked.")
                        break
                else:
                    st.warning("⚠️ Face not recognized.")
            else:
                st.error("❌ No face detected.")
