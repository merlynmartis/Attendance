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

# Page config
st.set_page_config(page_title="Face Attendance", layout="centered")
# Load and encode background image
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return encoded

# Set background
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

# Apply background
set_background("background.jpg")

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

st.markdown("""
    <style>
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #e9eef5;
            color: #1f1f1f;
        }

        /* Main App Background */
        .stApp {
            background-color: #f5f7fa;
            font-family: 'Segoe UI', sans-serif;
            color: #1f1f1f;
        }

        /* Block Container */
        .block-container {
            padding: 2rem 2rem;
        }

        /* Card Styling */
        .card {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 4px 12px rgba(0, 102, 204, 0.1);
            margin-bottom: 2rem;
        }

        /* Input Fields */
        input, textarea {
            background-color: #f0f4f8;
            border: 1px solid #a0c4ff;
            border-radius: 8px !important;
            padding: 0.5em;
            color: #1f1f1f;
        }

        /* File Uploader */
        .stFileUploader {
            background-color: #f0f4f8;
            padding: 1em;
            border-radius: 10px;
            border: 2px dashed #0077cc;
            color: #1f1f1f;
        }

        /* Buttons */
        .stButton > button {
            background-color: #0077cc;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.6em 1.2em;
            transition: background-color 0.3s ease;
        }

        .stButton > button:hover {
            background-color: #005fa3;
        }

        /* Headings */
        h1, h2, h3 {
            color: #004080;
        }

        /* DataFrame */
        .stDataFrame {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 1em;
        }

        /* Alerts */
        .stAlert {
            border-radius: 10px;
        }

        /* Markdown links */
        a {
            color: #0077cc;
        }
    </style>
""", unsafe_allow_html=True)

from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# Auto-refresh every 1 second (1000 ms)
st_autorefresh(interval=30000, limit=None, key="clock_refresh")

# Show the time
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.sidebar.markdown(f"🕒 **Current Time:** `{current_time}`")

# Initialize models
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


menu = st.sidebar.selectbox("Select Action",
                            ["Register Face", "Take Attendance", "Download Attendance Sheet", "Clear Attendance"])

admin_password = st.sidebar.text_input("🔐 Admin Password", type="password")

with st.container():
    if menu == "Register Face":
        st.markdown('<h3 style="text-align: center; color: #2b6777;"> Register New Face</h3>', unsafe_allow_html=True)
        name = st.text_input("Enter your name")
        uploaded_picture = st.file_uploader("Upload a picture", type=["jpg", "jpeg", "png"])


        registered_names = list(st.session_state.embeddings.keys())
        if registered_names:
            st.markdown("### 👥 Registered Users")
            for name in registered_names:
                st.markdown(f"- **{name}**")
        else:
            st.info("No users registered yet.")

        if uploaded_picture and name:
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


    elif menu == "Download Attendance Sheet":

        st.subheader("📥 Download Attendance Sheet")

        if admin_password == "secret123":  # Replace with your real password

            if st.session_state.attendance:

                df = pd.DataFrame(st.session_state.attendance)

                st.dataframe(df)

                csv = df.to_csv(index=False).encode('utf-8')

                st.download_button("Download CSV", csv, "attendance.csv", "text/csv")

            else:

                st.info("📝 No attendance data to download.")

        else:

            st.warning("🔒 Enter admin password in the sidebar to download attendance.")




    elif menu == "Clear Attendance":

        st.subheader("🧹 Clear Attendance Records")

        if admin_password == "secret123":  # Same password as above

            if st.button("Clear All Records"):
                st.session_state.attendance.clear()

                pd.DataFrame(columns=["Name", "Date", "Time"]).to_csv("attendance.csv", index=False)

                st.success("✅ All attendance records cleared.")

        else:

            st.warning("🔒 Enter admin password in the sidebar to clear records.")

    st.markdown('</div>', unsafe_allow_html=True)
