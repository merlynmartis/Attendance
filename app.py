import streamlit as st
import numpy as np
import cv2
import os
import torch
import random
from datetime import time
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
    try:
        today_sheet_name = date  # Format should be "YYYY-MM-DD"
        try:
            worksheet = gc.open_by_key(SHEET_ID).worksheet(today_sheet_name)
        except gspread.exceptions.WorksheetNotFound:
            worksheet = gc.open_by_key(SHEET_ID).add_worksheet(title=today_sheet_name, rows="1000", cols="3")
            worksheet.append_row(["Name", "Date", "Time"])  # Header

        worksheet.append_row([name, date, time])
        return True
    except Exception as e:
        st.error(f"❌ Failed to write to Google Sheet: {e}")
        return False


from datetime import datetime
import pytz

# Define IST timezone
ist = pytz.timezone('Asia/Kolkata')

def get_today_attendance():
    today_str = datetime.now(ist).strftime("%Y-%m-%d")
    try:
        worksheet = gc.open_by_key(SHEET_ID).worksheet(today_str)
        all_records = worksheet.get_all_records()
        return all_records
    except gspread.exceptions.WorksheetNotFound:
        return []  # No attendance yet for today
    except Exception as e:
        st.error(f"❌ Failed to read today's sheet: {e}")
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
    if isinstance(img, Image.Image):
        img_rgb = img.convert("RGB")
    elif isinstance(img, np.ndarray):
        if img.ndim == 3 and img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Invalid image array format")
        img_rgb = Image.fromarray(img_rgb)
    else:
        raise ValueError("Invalid image type passed to extract_face")

    face_tensor = mtcnn(img_rgb)
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
    st.subheader("📸 Take Attendance")

    # Load MTCNN and FaceNet
    mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, min_face_size=40)
    resnet = InceptionResnetV1(pretrained="vggface2").eval()

    # Load stored embeddings
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = {}
        if os.path.exists("data/registered_faces.npz"):
            data = np.load("data/registered_faces.npz")
            st.session_state.embeddings = {name: data[name] for name in data.files}

    # Get user location
    location = get_user_location()
    if location is None:
        st.warning("⚠️ Could not detect location.")  # You can remove this later
        location = INDIANA_LOCATION  # Fake location to allow webcam test

    if geodesic(location, INDIANA_LOCATION).km > LOCATION_RADIUS_KM:
        st.warning("You must be at Indiana Hospital & Heart Institute, Mangalore to mark attendance.")

    else:
        if st.button("🎥 Start Webcam"):
            st.info("Initializing webcam. Please wait...")
            cap = cv2.VideoCapture(0)
            stframe = st.empty()
            message_placeholder = st.empty()
            timeout = time.time() + 20  # 20-second timeout

            matched_name = None
            while time.time() < timeout:
                ret, frame = cap.read()
                if not ret:
                    message_placeholder.error("Failed to access webcam.")
                    break

                # Face detection and extraction
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_tensor = mtcnn(rgb)

                if face_tensor is not None:
                    emb = resnet(face_tensor.unsqueeze(0)).detach().numpy()
                    for name, db_emb in st.session_state.embeddings.items():
                        dist = np.linalg.norm(emb - db_emb)
                        if dist < 0.9:  # threshold
                            matched_name = name
                            break

                    if matched_name:
                        message_placeholder.success(f"✅ Face matched: {matched_name}")
                        break
                    else:
                        message_placeholder.warning("⚠️ Face not recognized. Try again.")

                stframe.image(frame, channels="BGR", caption="Capturing face...")

            cap.release()
            stframe.empty()

            # Mark attendance if matched
            if matched_name:
                now = datetime.now(ist)
                today_records = get_today_attendance()
                already_marked = any(rec["Name"] == matched_name for rec in today_records)

                if not already_marked:
                    success = append_attendance(
                        matched_name,
                        now.strftime("%Y-%m-%d"),
                        now.strftime("%H:%M:%S")
                    )
                    if success:
                        message_placeholder.success(f"🎉 Attendance marked for {matched_name}")
                    else:
                        message_placeholder.error("❌ Failed to update Google Sheets.")
                else:
                    message_placeholder.info(f"ℹ️ Attendance already marked for {matched_name} today.")


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
