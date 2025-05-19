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
import shutil

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

# Sidebar time
from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=30000, limit=None, key="clock_refresh")
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.sidebar.markdown(f"🕒 **Current Time:** `{current_time}`")

# Unified Select Action
menu = st.sidebar.selectbox(
    "Select Action",
    [
        "Register Face",
        "Take Attendance",
        "Download Attendance Sheet",
        "Clear Attendance",
        "View Registered Students"
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
    <p style="font-size: 20px; color: #2b6777; font-style: italic;">Presence, perfected through recognition.</p>
</div>
""", unsafe_allow_html=True)

# Register Face Page
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

# Take Attendance Page
elif menu == "Take Attendance":
    st.subheader("🧑‍💼 Take Attendance")
    uploaded = st.camera_input("Capture your face")

    if uploaded:
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
                    record = {"Name": name, "Date": now.strftime("%Y-%m-%d"), "Time": now.strftime("%H:%M:%S")}
                    if record not in st.session_state.attendance:
                        st.session_state.attendance.append(record)
                        st.success(f"🙌 Welcome {name}, attendance marked.")
                    else:
                        st.info(f"📌 Attendance already marked for {name}")
                    break
            else:
                st.warning("⚠️ Face not recognized. Try again.")
        else:
            st.error("❌ No face detected.")

# Download Attendance Page
elif menu == "Download Attendance Sheet":
    st.subheader("📥 Download Attendance Sheet")

    if admin_password == "secret123":
        if st.session_state.attendance:
            df = pd.DataFrame(st.session_state.attendance)
            st.dataframe(df)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "attendance.csv", "text/csv")
        else:
            st.info("📝 No attendance data to download.")
    else:
        st.warning("🔒 Enter admin password in the sidebar to download attendance.")

# Clear Attendance Page
elif menu == "Clear Attendance":
    st.subheader("🧹 Clear Attendance Records")

    if admin_password == "secret123":
        if st.button("Clear All Records"):
            st.session_state.attendance.clear()
            pd.DataFrame(columns=["Name", "Date", "Time"]).to_csv("attendance.csv", index=False)
            st.success("✅ All attendance records cleared.")
    else:
        st.warning("🔒 Enter admin password in the sidebar to clear records.")

# View Registered Students Page
elif menu == "View Registered Students":
    st.subheader("👥 Registered Students")
    if os.path.exists("data/registered_faces.npz"):
        with np.load("data/registered_faces.npz") as data:
            registered_names = list(data.files)
        if registered_names:
            for name in registered_names:
                st.markdown(f"- {name}")
            if admin_password == "secret123":
                if st.button("❌ Clear Registered Students"):
                    os.remove("data/registered_faces.npz")
                    st.session_state.embeddings = {}
                    st.success("✅ Registered students cleared.")
            else:
                st.warning("🔒 Enter correct admin password to clear registered students.")
        else:
            st.info("No students found in the data file.")
    else:
        st.info("📭 No registered students found.")
