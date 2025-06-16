import streamlit as st
import numpy as np
import cv2
import os
import torch
from datetime import datetime, time
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import pandas as pd
import base64
import requests
from geopy.distance import geodesic
import gspread
from google.oauth2.service_account import Credentials
import pytz

# Google Sheets Setup
SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
service_account_info = st.secrets["gcp_service_account"]
creds = Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
gc = gspread.authorize(creds)
SHEET_ID = '1lO0qt1EWZAwXjhRUOk19igYwI2rNyx5hLkG4wLyUkzc'

# Constants
INDIANA_LOCATION = (12.8697, 74.8426)
LOCATION_RADIUS_KM = 0.5
IST = pytz.timezone('Asia/Kolkata')

# Utility Functions
def get_user_location():
    try:
        loc = requests.get("https://ipinfo.io/json").json()['loc'].split(',')
        return float(loc[0]), float(loc[1])
    except:
        return None

def append_attendance(name, date, time):
    try:
        worksheet = None
        try:
            worksheet = gc.open_by_key(SHEET_ID).worksheet(date)
        except gspread.exceptions.WorksheetNotFound:
            worksheet = gc.open_by_key(SHEET_ID).add_worksheet(title=date, rows="1000", cols="3")
            worksheet.append_row(["Name", "Date", "Time"])
        worksheet.append_row([name, date, time])
        return True
    except Exception as e:
        st.error(f"❌ Google Sheet error: {e}")
        return False

def get_today_attendance():
    today = datetime.now(IST).strftime("%Y-%m-%d")
    try:
        worksheet = gc.open_by_key(SHEET_ID).worksheet(today)
        return worksheet.get_all_records()
    except gspread.exceptions.WorksheetNotFound:
        return []

# UI
st.set_page_config(page_title="Face Attendance", layout="centered")

# Models
device = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(image_size=160, margin=20, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Page menu
menu = st.sidebar.selectbox("Select Action", ["Register Face", "Take Attendance", "View Attendance Sheet", "View Registered Users"])
admin_password = st.sidebar.text_input("🔐 Admin Password", type="password")

# Embedding Storage
os.makedirs("data", exist_ok=True)
if "embeddings" not in st.session_state:
    st.session_state.embeddings = {}
    if os.path.exists("data/registered_faces.npz"):
        data = np.load("data/registered_faces.npz")
        st.session_state.embeddings = {k: data[k] for k in data.files}

# Register Face
if menu == "Register Face":
    st.header("👤 Register Face")
    name = st.text_input("Enter Name")
    img = st.file_uploader("Upload Face Image", type=["jpg", "png", "jpeg"])
    if name and img:
        try:
            image = Image.open(img).convert("RGB")  # ensure RGB
            face = mtcnn(image)

            if face is not None:
                emb = model(face.unsqueeze(0).to(device)).detach().cpu().numpy()[0]
                st.session_state.embeddings[name] = emb
                np.savez("data/registered_faces.npz", **st.session_state.embeddings)
                st.success(f"✅ {name} registered")
            else:
                st.warning("❌ No face detected.")
        except Exception as e:
            st.error(f"Error: {e}")

# Take Attendance
elif menu == "Take Attendance":
    st.header("📸 Take Attendance")
    location = get_user_location() or INDIANA_LOCATION
    if geodesic(location, INDIANA_LOCATION).km > LOCATION_RADIUS_KM:
        st.warning("You must be near Indiana Hospital, Mangalore.")
    else:
        start = st.button("🎥 Start Webcam")
        if start:
            cap = cv2.VideoCapture(0)
            stframe = st.empty()
            placeholder = st.empty()
            timeout = time.time() + 20
            matched = None
            while time.time() < timeout:
                ret, frame = cap.read()
                if not ret:
                    placeholder.error("Webcam access failed.")
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face = mtcnn(Image.fromarray(rgb))
                if face is not None:
                    emb = model(face.unsqueeze(0).to(device)).detach().cpu().numpy()[0]
                    for name, db_emb in st.session_state.embeddings.items():
                        if np.linalg.norm(emb - db_emb) < 0.9:
                            matched = name
                            break
                stframe.image(frame, channels="BGR", caption="Capturing...")
                if matched:
                    break
            cap.release()
            stframe.empty()

            if matched:
                now = datetime.now(IST)
                today = now.strftime("%Y-%m-%d")
                current_time = now.strftime("%H:%M:%S")
                already = any(r["Name"] == matched for r in get_today_attendance())
                if already:
                    placeholder.info(f"ℹ️ Attendance already marked for {matched}")
                else:
                    if append_attendance(matched, today, current_time):
                        placeholder.success(f"✅ Attendance marked for {matched}")
                    else:
                        placeholder.error("❌ Failed to mark attendance.")
            else:
                placeholder.warning("⚠️ No match found.")

# View Attendance
elif menu == "View Attendance Sheet":
    st.subheader("📅 Today's Attendance")
    records = get_today_attendance()
    if records:
        df = pd.DataFrame(records)
        st.dataframe(df)
    else:
        st.info("No records for today.")

# View Registered Users
elif menu == "View Registered Users":
    st.subheader("👥 Registered Users")
    if st.session_state.embeddings:
        for name in st.session_state.embeddings:
            st.markdown(f"- {name}")
        if admin_password == "secret123":
            if st.button("❌ Clear All"):
                os.remove("data/registered_faces.npz")
                st.session_state.embeddings.clear()
                st.success("✅ Cleared.")
        else:
            st.warning("🔒 Enter correct password to clear.")
    else:
        st.info("No registered users found.")
