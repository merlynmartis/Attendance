import streamlit as st
import numpy as np
import cv2
import os
import torch
from datetime import datetime
from PIL import Imageimport streamlit as st
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
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import io
from streamlit_autorefresh import st_autorefresh
from zoneinfo import ZoneInfo

# ---------- Page Config & Background ----------
st.set_page_config(page_title="Face Attendance", layout="centered")

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

# ---------- Google Auth & Drive Setup ----------
SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
service_account_info = st.secrets["gcp_service_account"]
creds = Credentials.from_service_account_info(service_account_info, scopes=SCOPES)

gc = gspread.authorize(creds)
SHEET_ID = '1lO0qt1EWZAwXjhRUOk19igYwI2rNyx5hLkG4wLyUkzc'  # Your Sheet ID
DRIVE_FOLDER_ID = "1jAjhyqMb8PEvaBy-hTBqBq02XaVSL9rk"

def get_drive_service():
    return build('drive', 'v3', credentials=creds)

def upload_file_to_drive(file_path, file_name, folder_id):
    service = get_drive_service()
    query = f"name='{file_name}' and '{folder_id}' in parents and trashed=false"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get('files', [])
    if files:
        file_id = files[0]['id']
        media = MediaFileUpload(file_path, resumable=True)
        service.files().update(fileId=file_id, media_body=media).execute()
    else:
        metadata = {'name': file_name, 'parents': [folder_id]}
        media = MediaFileUpload(file_path, resumable=True)
        service.files().create(body=metadata, media_body=media).execute()

def download_file_from_drive(file_name, folder_id, dest_path):
    service = get_drive_service()
    query = f"name='{file_name}' and '{folder_id}' in parents and trashed=false"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get('files', [])
    if not files:
        return False
    file_id = files[0]['id']
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(dest_path, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return True

# ---------- Constants ----------
INDIANA_LOCATION = (12.8697, 74.8426)
LOCATION_RADIUS_KM = 0.5
os.makedirs("data", exist_ok=True)
registered_path = "data/registered_faces.npz"

# ---------- Load Registered Embeddings from Drive ----------
if download_file_from_drive("registered_faces.npz", DRIVE_FOLDER_ID, registered_path):
    data = np.load(registered_path)
    st.session_state.embeddings = {n: data[n] for n in data.files}
else:
    st.session_state.embeddings = {}
st.session_state.attendance = []

# ---------- Face Recognition Setup ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(image_size=160, margin=20, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ---------- Utility Functions ----------
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

# ---------- UI and Logic ----------
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

st_autorefresh(interval=60000, limit=None, key="clock_refresh")

ist = ZoneInfo("Asia/Kolkata")
current_time = datetime.now(ist).strftime("%Y-%m-%d %H:%M")
st.sidebar.markdown(f"🕒 **Current Time (IST):** `{current_time}`")

menu = st.sidebar.selectbox("Menu", [
    "Register Face",
    "Take Attendance",
    "View Attendance Sheet",
    "View Registered Users"
])
admin_password = st.sidebar.text_input("🔐 Admin Password", type="password")

# ---------- Register Face ----------
if menu == "Register Face":
    st.markdown('<h3 style="text-align: center; color: #2b6777;"> Register New Face</h3>', unsafe_allow_html=True)
    name = st.text_input("Enter your name")
    uploaded = st.file_uploader("Upload a picture", type=["jpg", "jpeg", "png"])
    if uploaded and name:
        img = np.array(Image.open(uploaded))
        face_tensor = extract_face(img)
        if face_tensor is not None:
            emb = get_embedding(face_tensor)
            st.session_state.embeddings[name] = emb
            np.savez(registered_path, **st.session_state.embeddings)
            upload_file_to_drive(registered_path, "registered_faces.npz", DRIVE_FOLDER_ID)
            st.success(f"✅ Registered {name}")
        else:
            st.error("❌ No face detected.")

# ---------- Take Attendance ----------
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

# ---------- View Attendance Sheet ----------
elif menu == "View Attendance Sheet":
    st.subheader("📅 Today's Attendance")
    today_records = get_today_attendance()
    if today_records:
        st.dataframe(pd.DataFrame(today_records))
    else:
        st.info("📭 No attendance found for today.")

# ---------- View Registered Users ----------
elif menu == "View Registered Users":
    st.subheader("👥 Registered Users")
    if os.path.exists(registered_path):
        with np.load(registered_path) as data:
            names = list(data.files)
        if names:
            st.markdown("### Registered:")
            for name in names:
                st.markdown(f"- {name}")
            if admin_password == "secret123":
                if st.button("❌ Clear Registered Users"):
                    os.remove(registered_path)
                    st.session_state.embeddings = {}
                    upload_file_to_drive(registered_path, "registered_faces.npz", DRIVE_FOLDER_ID)
                    st.success("✅ Cleared all users.")
            else:
                st.warning("🔒 Enter correct admin password to clear users.")
        else:
            st.info("📭 No users found.")
    else:
        st.info("📭 No registered users yet.")

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


from streamlit_autorefresh import st_autorefresh
from zoneinfo import ZoneInfo

# Refresh clock every 60 seconds
st_autorefresh(interval=60000, limit=None, key="clock_refresh")

# Set IST timezone
ist = ZoneInfo("Asia/Kolkata")
current_time = datetime.now(ist).strftime("%Y-%m-%d %H:%M")

# Display time in sidebar
st.sidebar.markdown(f"🕒 **Current Time (IST):** `{current_time}`")
menu = st.sidebar.selectbox("Menu", [
    "Register Face",
    "Take Attendance",
    "View Attendance Sheet",
    "View Registered Users"
])
admin_password = st.sidebar.text_input("🔐 Admin Password", type="password")

# Register Face
if menu == "Register Face":
    st.markdown('<h3 style="text-align: center; color: #2b6777;"> Register New Face</h3>', unsafe_allow_html=True)
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
