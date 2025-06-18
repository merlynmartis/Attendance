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
from streamlit_js_eval import streamlit_js_eval

# --- Page Config & Background ---
st.set_page_config(page_title="Presencia – Location-Aware Attendance", layout="centered")
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(f"""
        <style>.stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}</style>
    """, unsafe_allow_html=True)
set_background("background.jpg")

# --- Config Variables ---
HOSPITAL_LAT = 12.8880
HOSPITAL_LON = 74.8426
ALLOWED_RADIUS_KM = 0.5

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))

# --- Google Auth & File Storage ---
# (Same as before – credentials, upload/download)

# --- Face Recognition Setup ---
# (Same as before – MTCNN, model, extract_face, get_embedding, is_match)

# --- Attendance Utilities & Sheets Logic ---
# (Same as before – append_attendance, get_today_attendance)

# --- UI & Logic ---
st.title("🎯 Presencia – Face + Location Attendance")
st_autorefresh(interval=60000, key="clock")

# Get GPS coordinates via browser
location = streamlit_js_eval(js_expressions="navigator.geolocation.getCurrentPosition((p) => p.coords)", key="gps")

if not location or location.get("latitude") is None:
    st.warning("⚠ Please allow browser location permissions to proceed.")
    st.stop()

lat, lon = location["latitude"], location["longitude"]
st.success(f"📍 GPS location: {lat:.6f}, {lon:.6f}")

distance = haversine(lat, lon, HOSPITAL_LAT, HOSPITAL_LON)
st.info(f"📏 Distance from Indiana Hospital: **{distance:.2f} km**")

if distance > ALLOWED_RADIUS_KM:
    st.error("🚫 You are NOT within the permitted 0.5 km radius.")
    st.stop()

menu = st.sidebar.selectbox("Menu", ["Register Face", "Take Attendance", "View Attendance", "Registered Users"])
ist = ZoneInfo("Asia/Kolkata")
current_time = datetime.now(ist).strftime("%Y-%m-%d %H:%M")
st.sidebar.markdown(f"🕒 **IST Time:** {current_time}")

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
    captured = st.camera_input("Take your selfie for attendance")
    if captured:
        img = cv2.imdecode(np.frombuffer(captured.read(), np.uint8), cv2.IMREAD_COLOR)
        face = extract_face(img)
        if face is not None:
            emb = get_embedding(face)
            for name, known in st.session_state.embeddings.items():
                if is_match(known, emb):
                    now = datetime.now(ist)
                    d, t = now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")
                    rec = {"Name": name, "Date": d, "Time": t}
                    if rec not in st.session_state.attendance:
                        st.session_state.attendance.append(rec)
                        append_attendance(name, d, t)
                        st.success(f"✅ {name}, attendance marked!")
                    else:
                        st.info("ℹ Already marked today.")
                    break
            else:
                st.warning("⚠ Face not recognized.")
        else:
            st.error("❌ No face detected. Try again.")

elif menu == "View Attendance":
    df = pd.DataFrame(get_today_attendance())
    st.subheader("📅 Today's Attendance")
    if not df.empty:
        st.dataframe(df)
    else:
        st.info("📭 No attendance today.")

elif menu == "Registered Users":
    st.subheader("👥 Registered Users")
    names = list(st.session_state.embeddings.keys())
    if names:
        for n in names:
            st.markdown(f"- {n}")
    else:
        st.info("📭 No registered users yet.")
