import streamlit as st
import torch
import numpy as np
import os
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load models
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Create directory for data
os.makedirs("data", exist_ok=True)

# Load stored embeddings
def load_embeddings():
    embeddings = {}
    if os.path.exists("data/registered_faces.npz"):
        data = np.load("data/registered_faces.npz", allow_pickle=True)
        for key in data.files:
            embeddings[key] = data[key]
    return embeddings

# Save embeddings
def save_embeddings(embeddings):
    np.savez("data/registered_faces.npz", **embeddings)

# Extract face tensor
def extract_face(image):
    try:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")
        else:
            image = image.convert("RGB")
        face = mtcnn(image)
        if face is not None:
            return face.unsqueeze(0).to(device)
        else:
            return None
    except Exception as e:
        st.error(f"❌ Failed to process image: {e}")
        return None

# Get embedding
def get_embedding(face_tensor):
    with torch.no_grad():
        embedding = resnet(face_tensor)
    return embedding

# Compare embeddings (cosine similarity)
def match_embedding(new_embedding, stored_embeddings):
    threshold = 0.6
    for name, emb in stored_embeddings.items():
        emb_tensor = torch.tensor(emb).to(device)
        sim = torch.nn.functional.cosine_similarity(new_embedding, emb_tensor.unsqueeze(0)).item()
        if sim > threshold:
            return name, sim
    return None, None

# Streamlit app
st.title("📸 Face Recognition Attendance System")

menu = st.sidebar.selectbox("Menu", ["Register Face", "Take Attendance"])

# Initialize session state for embeddings
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = load_embeddings()

# Register Face
if menu == "Register Face":
    st.markdown('<h3 style="text-align: center; color: #2b6777;">Register New Face</h3>', unsafe_allow_html=True)
    name = st.text_input("Enter your name")
    uploaded_picture = st.file_uploader("Upload a picture", type=["jpg", "jpeg", "png"])

    if uploaded_picture and name:
        image = Image.open(uploaded_picture)
        face_tensor = extract_face(image)

        if face_tensor is not None:
            embedding = get_embedding(face_tensor).cpu().detach().numpy()
            st.session_state.embeddings[name] = embedding
            save_embeddings(st.session_state.embeddings)
            st.success(f"✅ Face registered for {name}")
        else:
            st.error("❌ No face detected. Try a clearer image.")

# Take Attendance
elif menu == "Take Attendance":
    st.markdown('<h3 style="text-align: center; color: #2b6777;">Take Attendance</h3>', unsafe_allow_html=True)
    uploaded_picture = st.file_uploader("Upload a picture for attendance", type=["jpg", "jpeg", "png"])

    if uploaded_picture:
        image = Image.open(uploaded_picture)
        face_tensor = extract_face(image)

        if face_tensor is not None:
            new_embedding = get_embedding(face_tensor)
            matched_name, similarity = match_embedding(new_embedding, st.session_state.embeddings)
            if matched_name:
                st.success(f"✅ Attendance marked for {matched_name} (Similarity: {similarity:.2f})")
            else:
                st.warning("⚠️ No matching face found.")
        else:
            st.error("❌ No face detected.")
