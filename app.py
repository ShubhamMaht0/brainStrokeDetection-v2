import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
import sqlite3
import hashlib
from datetime import datetime
import pandas as pd


# Initialize database connection
conn = sqlite3.connect("patients.db")
c = conn.cursor()

# Create tables if not exist
def init_db():
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        username TEXT UNIQUE,
        password TEXT
    )
    ''')
    c.execute('''
    CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY,
        name TEXT,
        age INTEGER,
        gender TEXT
    )
    ''')
    c.execute('''
    CREATE TABLE IF NOT EXISTS stroke_records (
        id INTEGER PRIMARY KEY,
        patient_id INTEGER,
        date TEXT,
        image BLOB,
        result TEXT,
        FOREIGN KEY (patient_id) REFERENCES patients (id)
    )
    ''')
    conn.commit()

# Register a new user
def register_user(username, password):
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password_hash))
        conn.commit()
        st.success("User registered!")
    except sqlite3.IntegrityError:
        st.error("Username already exists")

# Validate user login
def validate_user(username, password):
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password_hash))
    return c.fetchone() is not None

# Register a new patient
def register_patient(name, age, gender):
    c.execute("INSERT INTO patients (name, age, gender) VALUES (?, ?, ?)", (name, age, gender))
    conn.commit()

# Save a stroke analysis record
def save_stroke_record(patient_id, image, result):
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO stroke_records (patient_id, date, image, result) VALUES (?, ?, ?, ?)",
              (patient_id, date, image, result))
    conn.commit()

# Fetch patient's history
def get_patient_history(patient_id):
    c.execute("SELECT date, result FROM stroke_records WHERE patient_id = ?", (patient_id,))
    records = c.fetchall()
    return pd.DataFrame(records, columns=["Date", "Result"])

# Sample prediction function for stroke detection
def predict_stroke(image):
    # Placeholder prediction logic
    return "Stroke Detected" if image else "No Stroke Detected"

# Load your pre-trained model (ensure it's properly compiled)
def load_model():
    model = tf.keras.models.load_model('brain_stroke_model.keras')
    return model

# Function to preprocess the image step-by-step
def preprocess_image(image):
    # Stage 1: Grayscale Conversion
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Stage 2: Blurring (Gaussian Blur)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Stage 3: Edge Detection (Canny)
    edges_image = cv2.Canny(blurred_image, 100, 200)

    return gray_image, blurred_image, edges_image

# Function to make predictions
def predict(image):
    # Preprocess the image as needed
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0) / 255.0  # Rescale image
    prediction = model.predict(image)
    return prediction

# Load model
model = load_model()

# Main application
def main():
    st.title("Brain Stroke Detection App")
    init_db()
    
    # Sidebar for Login or Registration
    st.sidebar.title("User Login/Registration")
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    
    if st.session_state['logged_in']:
        st.sidebar.write("Welcome!")
        if st.sidebar.button("Logout"):
            st.session_state['logged_in'] = False
    else:
        choice = st.sidebar.selectbox("Choose Action", ["Login", "Register"])
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")

        if choice == "Login":
            if st.sidebar.button("Login"):
                if validate_user(username, password):
                    st.session_state['logged_in'] = True
                    st.sidebar.success("Logged in as {}".format(username))
                else:
                    st.sidebar.error("Invalid credentials")
        elif choice == "Register":
            if st.sidebar.button("Register"):
                register_user(username, password)

    # Main app content after login
    if st.session_state['logged_in']:
        st.subheader("Register a New Patient")
        name = st.text_input("Patient Name")
        age = st.number_input("Age", min_value=0)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        if st.button("Register Patient"):
            register_patient(name, age, gender)
            st.success("Patient registered!")

        # Patient Selection for Stroke Analysis
        st.subheader("Stroke Analysis")
        c.execute("SELECT id, name FROM patients")
        patients = c.fetchall()
        patient_id = st.selectbox("Select Patient", [p[0] for p in patients], format_func=lambda x: dict(patients)[x])        

        # Upload image and predict stroke
        uploaded_file = st.file_uploader("Upload Brain Scan Image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            # Read uploaded image
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

            # Preprocess the image step-by-step
            gray_image, blurred_image, edges_image = preprocess_image(image)

            # Display the process in columns
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.image(image, channels="BGR", caption="Original Image")

            with col2:
                st.image(gray_image, caption="Grayscale Image", use_column_width=True)

            with col3:
                st.image(blurred_image, caption="Blurred Image", use_column_width=True)

            with col4:
                st.image(edges_image, caption="Edge Detection", use_column_width=True)

            # Final Prediction (using original image processed for the model)
            prediction = predict(image)
            st.write(f"Prediction: {'Stroke' if prediction[0] > 0.5 else 'No Stroke'}")
            st.write(prediction[0])
            if st.button("Save Result"):
                save_stroke_record(patient_id, uploaded_file.getvalue(), 'Stroke' if prediction[0] > 0.5 else 'No Stroke')
                st.success("Record saved!")

        # Patient History
        st.subheader("Patient History")
        if patient_id:
            history_df = get_patient_history(patient_id)
            st.write("Historical Records for", dict(patients)[patient_id])
            st.dataframe(history_df)

if __name__ == "__main__":
    main()



