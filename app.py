import streamlit as st
import numpy as np
import cv2
import os
import torch
import math
import requests
from datetime import datetime
from PIL import Image
import base64
from zoneinfo import ZoneInfo
import pandas as pd

import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

from facenet_pytorch import MTCNN, InceptionResnetV1
from streamlit_autorefresh import st_autorefresh

# --- Configuration ---
st.set_page_config(page_title="Presencia - Face Attendance", layout="centered")

# Indiana Hospital Mangalore Coordinates
HOSPITAL_LAT = 12.8880
HOSPITAL_LON = 74.8426
ALLOWED_RADIUS_KM = 0.5  # 500 meters

# --- Style & Background ---
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
    """, unsafe_allow_html=True)

set_background("background.jpg")

# --- Google Sheets & Drive Setup ---
SCOPES = ['https://www.googleapis.com/auth/spreadsheets',
          'https://www.googleapis.com/auth/drive']
creds = Credentials.from_service_account_info(
    st.secrets["gcp_service_account"], scopes=SCOPES
)
gc = gspread.authorize(creds)
SHEET_ID = '1lO0qt1EWZAwXjhRUOk19igYwI2rNyx5hLkG4wLyUkzc'
DRIVE_FOLDER_ID = "1jAjhyqMb8PEvaBy-hTBqBq02XaVSL9rk"
REGISTERED_PATH = "data/registered_faces.npz"

def get_drive_service():
    return build('drive', 'v3', credentials=creds)

def upload_file_to_drive(file_path, file_name):
    service = get_drive_service()
    query = f"name='{file_name}' and '{DRIVE_FOLDER_ID}' in parents and trashed=false"
    files = service.files().list(q=query, fields="files(id)").execute().get('files', [])
    media = MediaFileUpload(file_path, resumable=False)
    if files:
        service.files().update(fileId=files[0]['id'], media_body=media).execute()
    else:
        service.files().create(
            body={'name': file_name, 'parents': [DRIVE_FOLDER_ID]},
            media_body=media
        ).execute()

def download_file_from_drive(file_name, dest_path):
    service = get_drive_service()
    query = f"name='{file_name}' and '{DRIVE_FOLDER_ID}' in parents and trashed=false"
    files = service.files().list(q=query, fields="files(id)").execute().get('files', [])
    if not files:
        return False
    request = service.files().get_media(fileId=files[0]['id'])
    with open(dest_path, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return True

# --- Session State Init ---
if "embeddings" not in st.session_state:
    os.makedirs("data", exist_ok=True)
    if download_file_from_drive("registered_faces.npz", REGISTERED_PATH):
        data = np.load(REGISTERED_PATH)
        st.session_state.embeddings = {n: data[n] for n in data.files}
    else:
        st.session_state.embeddings = {}

if "attendance" not in st.session_state:
    st.session_state.attendance = []

# --- Face Recognition Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(image_size=160, margin=20, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def extract_face(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_tensor = mtcnn(Image.fromarray(img_rgb))
    return face_tensor.unsqueeze(0).to(device) if face_tensor is not None else None

def get_embedding(face_tensor):
    with torch.no_grad():
        return model(face_tensor)[0].cpu().numpy()

def is_match(known, candidate, thresh=0.9):
    return np.linalg.norm(known - candidate) < thresh

# --- Attendance Logging ---
def append_attendance(name, date, time):
    try:
        worksheet = gc.open_by_key(SHEET_ID).worksheet(date)
    except gspread.exceptions.WorksheetNotFound:
        worksheet = gc.open_by_key(SHEET_ID).add_worksheet(title=date, rows="1000", cols="3")
        worksheet.append_row(["Name", "Date", "Time"])
    worksheet.append_row([name, date, time])

def get_today_attendance():
    date = datetime.now().strftime("%Y-%m-%d")
    try:
        return gc.open_by_key(SHEET_ID).worksheet(date).get_all_records()
    except:
        return []

# --- Utility: IP Geolocation + Distance Check ---
def get_ip_location():
    try:
        res = requests.get("https://ipinfo.io/json", timeout=5)
        data = res.json()
        if 'loc' in data:
            lat, lon = map(float, data['loc'].split(','))
            return lat, lon
    except:
        return None, None

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2)**2 +
        math.cos(math.radians(lat1)) *
        math.cos(math.radians(lat2)) *
        math.sin(dlon / 2)**2
    )
    c = 2 * math.asin(math.sqrt(a))
    return R * c

# --- UI Layout ---
st.markdown("""
<div style="
  background-color:#ecf6f7;
  padding:1.5rem;
  border-radius:15px;
  text-align:center;
  box-shadow:0 4px 10px rgba(43,103,119,0.15);">
  <h2 style="color:#2b6777; margin-bottom:0.5rem">
    Presencia - A Face Attendance System
  </h2>
  <p style="font-size:20px; color:#2b6777; font-style:italic;">
    Look once. You're marked present.
  </p>
</div>
""", unsafe_allow_html=True)

st_autorefresh(interval=60000, key="clock_refresh")
ist = ZoneInfo("Asia/Kolkata")
current_time = datetime.now(ist).strftime("%Y-%m-%d %H:%M")
st.sidebar.markdown(f"🕒 *Current Time (IST):* {current_time}")

menu = st.sidebar.selectbox("Menu", ["Register Face", "Take Attendance", "View Attendance Sheet", "View Registered Users"])

# --- Menu Actions ---
if menu == "Register Face":
    st.subheader("Register New Face")
    name = st.text_input("Enter your name")
    uploaded = st.file_uploader("Upload a picture", type=["jpg","jpeg","png"])

    if uploaded and name:
        img = np.array(Image.open(uploaded))
        face = extract_face(img)
        if face is not None:
            st.session_state.embeddings[name] = get_embedding(face)
            np.savez(REGISTERED_PATH, **st.session_state.embeddings)
            upload_file_to_drive(REGISTERED_PATH, "registered_faces.npz")
            st.success(f"✅ Registered {name}")
        else:
            st.error("❌ No face detected.")

elif menu == "Take Attendance":
    st.subheader("📸 Take Attendance")

    # ✅ Location Restriction Step
    lat, lon = get_ip_location()
    if lat is None or lon is None:
        st.error("❌ Unable to retrieve location. Attendance denied.")
        st.stop()

    distance = haversine(lat, lon, HOSPITAL_LAT, HOSPITAL_LON)
    st.info(f"📍 Distance from hospital: **{distance:.2f} km**")
    if distance > ALLOWED_RADIUS_KM:
        st.error("🚫 You are outside the allowed 0.5 km range.")
        st.stop()

    captured = st.camera_input("Take your photo")
    if captured:
        file_bytes = np.asarray(bytearray(captured.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        face = extract_face(img)
        if face is not None:
            emb = get_embedding(face)
            for name, known_emb in st.session_state.embeddings.items():
                if is_match(known_emb, emb):
                    now = datetime.now(ist)
                    date_str, time_str = now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")
                    rec = {"Name": name, "Date": date_str, "Time": time_str}
                    if rec not in st.session_state.attendance:
                        st.session_state.attendance.append(rec)
                        append_attendance(name, date_str, time_str)
                        st.success(f"✅ Attendance marked for {name}")
                    else:
                        st.info("ℹ Already marked today.")
                    break
            else:
                st.warning("⚠ Face not recognized.")
        else:
            st.error("❌ No face detected.")

elif menu == "View Attendance Sheet":
    st.subheader("📅 Today's Attendance")
    df = pd.DataFrame(get_today_attendance())
    if not df.empty: st.dataframe(df)
    else: st.info("📭 No attendance recorded today.")

elif menu == "View Registered Users":
    st.subheader("👥 Registered Users")
    if os.path.exists(REGISTERED_PATH):
        with np.load(REGISTERED_PATH) as data:
            names = data.files
        if names:
            for name in names:
                st.markdown(f"- {name}")
        else:
            st.info("📭 No registered users yet.")
    else:
        st.info("📭 No registered users yet.")
