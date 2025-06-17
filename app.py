import streamlit as st
import numpy as np
import cv2
import os
import torch
from datetime import datetime
from PIL import Image
import base64
import requests
import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from facenet_pytorch import MTCNN, InceptionResnetV1
from streamlit_autorefresh import st_autorefresh
from zoneinfo import ZoneInfo
import pandas as pd
import io

# --------------- Config ----------------
st.set_page_config(page_title="Presencia - Face Attendance", layout="centered")

# Set Background
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

# --------------- Google Sheets & Drive ----------------
SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
service_account_info = st.secrets["gcp_service_account"]
creds = Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
gc = gspread.authorize(creds)
SHEET_ID = '1lO0qt1EWZAwXjhRUOk19igYwI2rNyx5hLkG4wLyUkzc'
DRIVE_FOLDER_ID = "1jAjhyqMb8PEvaBy-hTBqBq02XaVSL9rk"
REGISTERED_PATH = "data/registered_faces.npz"

def get_drive_service():
    return build('drive', 'v3', credentials=creds)

def upload_file_to_drive(file_path, file_name):
    service = get_drive_service()
    query = f"name='{file_name}' and '{DRIVE_FOLDER_ID}' in parents and trashed=false"
    results = service.files().list(q=query, fields="files(id)").execute()
    files = results.get("files", [])
    media = MediaFileUpload(file_path, resumable=False)  # ← resumable=False fixes many upload errors
    if files:
        service.files().update(fileId=files[0]['id'], media_body=media).execute()
    else:
        service.files().create(body={'name': file_name, 'parents': [DRIVE_FOLDER_ID]}, media_body=media).execute()


def download_file_from_drive(file_name, dest_path):
    service = get_drive_service()
    query = f"name='{file_name}' and '{DRIVE_FOLDER_ID}' in parents and trashed=false"
    files = service.files().list(q=query, fields="files(id)").execute().get('files', [])
    if not files: return False
    request = service.files().get_media(fileId=files[0]['id'])
    with open(dest_path, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return True

import streamlit.components.v1 as components
import math

# Indiana Hospital Location (Mangalore)
HOSPITAL_LAT = 12.8699
HOSPITAL_LON = 74.8428
ALLOWED_RADIUS_METERS = 50  # meters


# --------------- Session State Init ----------------
if "embeddings" not in st.session_state:
    os.makedirs("data", exist_ok=True)
    if download_file_from_drive("registered_faces.npz", REGISTERED_PATH):
        data = np.load(REGISTERED_PATH)
        st.session_state.embeddings = {n: data[n] for n in data.files}
    else:
        st.session_state.embeddings = {}

if "attendance" not in st.session_state:
    st.session_state.attendance = []

# --------------- Face Recognition ----------------
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


# --------------- Attendance ----------------
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
        return gc.open_by_key(SHEET_ID).worksheet(date).get_all_records()
    except:
        return []

# --------------- UI ----------------
st.markdown("""
<div style="background-color: #ecf6f7; padding: 1.5rem; border-radius: 15px; text-align: center; box-shadow: 0 4px 10px rgba(43, 103, 119, 0.15);">
    <h2 style="color: #2b6777; margin-bottom: 0.5rem;">Presencia - A Face Attendance System</h2>
    <p style="font-size: 20px; color: #2b6777; font-style: italic;">Look once. You're marked present.</p>
</div>
""", unsafe_allow_html=True)

st_autorefresh(interval=60000, key="clock_refresh")

ist = ZoneInfo("Asia/Kolkata")
current_time = datetime.now(ist).strftime("%Y-%m-%d %H:%M")
st.sidebar.markdown(f"🕒 **Current Time (IST):** `{current_time}`")

menu = st.sidebar.selectbox("Menu", ["Register Face", "Take Attendance", "View Attendance Sheet", "View Registered Users"])

# --------------- Functional Menus ----------------

if menu == "Register Face":
    st.subheader("Register New Face")
    name = st.text_input("Enter your name")
    uploaded = st.file_uploader("Upload a picture", type=["jpg", "jpeg", "png"])
    if uploaded and name:
        img = np.array(Image.open(uploaded))
        face_tensor = extract_face(img)
        if face_tensor is not None:
            st.session_state.embeddings[name] = get_embedding(face_tensor)
            np.savez(REGISTERED_PATH, **st.session_state.embeddings)
            upload_file_to_drive(REGISTERED_PATH, "registered_faces.npz")
            st.success(f"✅ Registered {name}")
        else:
            st.error("❌ No face detected.")

elif menu == "Take Attendance":

    st.subheader("📸 Take Attendance")

    # ---- Location Logic ----
    st.session_state["location_checked"] = True
    query_params = st.query_params

    if "lat" not in query_params or "lon" not in query_params:
        st.markdown("""
            <script>
            navigator.geolocation.getCurrentPosition(
                function(pos) {
                    const lat = pos.coords.latitude;
                    const lon = pos.coords.longitude;
                    const newUrl = window.location.protocol + "//" + window.location.host + window.location.pathname + "?lat=" + lat + "&lon=" + lon;
                    window.location.replace(newUrl);
                },
                function(err) {
                    alert("Location access is required to mark attendance.");
                }
            );
            </script>
        """, unsafe_allow_html=True)
        st.info("📍 Detecting your location... please allow location access.")
        st.stop()

    # ---- Location Check ----
    def haversine(lat1, lon1, lat2, lon2):
        from math import radians, sin, cos, sqrt, atan2
        R = 6371000  # Earth radius in meters
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c

    user_lat = float(query_params["lat"][0])
    user_lon = float(query_params["lon"][0])
    hospital_lat, hospital_lon = 12.8692, 74.8560  # Indiana Hospital & Heart Institute

    distance = haversine(user_lat, user_lon, hospital_lat, hospital_lon)
    if distance > 100:  # adjust this to 50 or 25 if needed
        st.error("🚫 You must be inside Indiana Hospital & Heart Institute to mark attendance.")
        st.stop()

    # ---- Proceed with Face Attendance ----
    captured = st.camera_input("Take your photo")
    if captured:
        file_bytes = np.asarray(bytearray(captured.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        face_tensor = extract_face(img)
        if face_tensor is not None:
            emb = get_embedding(face_tensor)
            for name, known_emb in st.session_state.embeddings.items():
                if is_match(known_emb, emb):
                    now = datetime.now()
                    date, time = now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")
                    record = {"Name": name, "Date": date, "Time": time}
                    if record not in st.session_state.attendance:
                        st.session_state.attendance.append(record)
                        append_attendance(name, date, time)
                        st.success(f"✅ Attendance marked for {name}")
                    else:
                        st.info("ℹ️ Already marked today.")
                    break
            else:
                st.warning("⚠️ Face not recognized.")
        else:
            st.error("❌ No face detected.")


    # --- Continue only if location is verified ---
    if st.session_state.location_ok:
        captured = st.camera_input("Take your photo")
        if captured:
            file_bytes = np.asarray(bytearray(captured.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            face_tensor = extract_face(img)
            if face_tensor is not None:
                emb = get_embedding(face_tensor)
                for name, known_emb in st.session_state.embeddings.items():
                    if is_match(known_emb, emb):
                        now = datetime.now()
                        date, time = now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")
                        record = {"Name": name, "Date": date, "Time": time}
                        if record not in st.session_state.attendance:
                            st.session_state.attendance.append(record)
                            append_attendance(name, date, time)
                            st.success(f"✅ Attendance marked for {name}")
                        else:
                            st.info("ℹ️ Already marked today.")
                        break
                else:
                    st.warning("⚠️ Face not recognized.")
            else:
                st.error("❌ No face detected.")

elif menu == "View Registered Users":
    st.subheader("👥 Registered Users")
    if os.path.exists(REGISTERED_PATH):
        with np.load(REGISTERED_PATH) as data:
            names = list(data.files)
        if names:
            for name in names:
                st.markdown(f"- {name}")
        else:
            st.info("📭 No users found.")
    else:
        st.info("📭 No registered users yet.")
