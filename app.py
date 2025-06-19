elif menu == "Take Attendance":
    from streamlit_folium import st_folium
    import folium
    from math import radians, cos, sin, asin, sqrt

    # Indiana Hospital coordinates
    HOSPITAL_LAT = 12.8682
    HOSPITAL_LON = 74.8661
    ALLOWED_RADIUS_METERS = 500

    def haversine(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * asin(sqrt(a))
        r = 6371000  # Earth radius in meters
        return c * r

    st.subheader("📍 Verifying your location...")
    m = folium.Map(location=[HOSPITAL_LAT, HOSPITAL_LON], zoom_start=17)
    folium.Circle(
        location=[HOSPITAL_LAT, HOSPITAL_LON],
        radius=ALLOWED_RADIUS_METERS,
        color="green", fill=True, fill_opacity=0.3
    ).add_to(m)
    folium.Marker(
        [HOSPITAL_LAT, HOSPITAL_LON],
        tooltip="Indiana Hospital"
    ).add_to(m)
    folium.plugins.LocateControl(
        auto_start=True,
        keepCurrentZoomLevel=True,
        showPopup=True,
        strings={"title": "Tap the blue dot to confirm your location"},
        locateOptions={"enableHighAccuracy": True}
    ).add_to(m)

    location_data = st_folium(m, width=700, height=500)
    lat = lon = None
    if location_data:
        if location_data.get("last_clicked"):
            lat = location_data["last_clicked"]["lat"]
            lon = location_data["last_clicked"]["lng"]
            st.info("📌 You clicked on the map to confirm location.")
        elif location_data.get("location"):
            lat = location_data["location"]["lat"]
            lon = location_data["location"]["lng"]
            st.info("📍 Using browser GPS location (auto-detected).")

    if lat and lon:
        st.success(f"📡 Your Location: {lat}, {lon}")
        distance = haversine(lat, lon, HOSPITAL_LAT, HOSPITAL_LON)
        st.info(f"📏 Distance from Hospital: {int(distance)} meters")
        if distance > ALLOWED_RADIUS_METERS:
            st.error("❌ You are outside the allowed attendance zone.")
            st.stop()
    else:
        st.warning("📍 Location not available. Please tap the blue dot or enable GPS.")
        st.stop()

    # Proceed with attendance capture if location is valid
    st.subheader("📸 Now take your photo")
    captured = st.camera_input("Take your photo")
    if captured:
        file_bytes = np.asarray(bytearray(captured.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        face_tensor = extract_face(img)
        if face_tensor is not None:
            emb = get_embedding(face_tensor)
            for name, known_emb in st.session_state.embeddings.items():
                if is_match(known_emb, emb):
                    now = datetime.now(ZoneInfo("Asia/Kolkata"))
                    date, time = now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")
                    record = {"Name": name, "Date": date, "Time": time}
                    if record not in st.session_state.attendance:
                        st.session_state.attendance.append(record)
                        append_attendance(name, date, time)
                        st.success(f"✅ Attendance marked for {name}")
                    else:
                        st.info("ℹ Already marked today.")
                    break
            else:
                st.warning("⚠ Face not recognized.")
        else:
            st.error("❌ No face detected.")
