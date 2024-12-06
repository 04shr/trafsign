import streamlit as st
import cv2
import os
import numpy as np

# Function to load stored traffic sign images
def load_traffic_signs(folder_path):
    traffic_signs = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            label = os.path.splitext(file_name)[0]
            label = label.replace('_', ' ').title()  # Capitalize first letter and replace underscores with spaces
            img_path = os.path.join(folder_path, file_name)
            img = cv2.imread(img_path)  # Read in color
            traffic_signs[label] = img
    return traffic_signs

# Function to find the best match for the uploaded image based on histogram comparison
def find_best_match(uploaded_img, traffic_signs):
    # Convert the uploaded image to HSV (Hue, Saturation, Value) for better comparison
    uploaded_img_hsv = cv2.cvtColor(uploaded_img, cv2.COLOR_BGR2HSV)
    best_match = None
    best_score = -1

    for label, stored_img in traffic_signs.items():
        # Convert the stored image to HSV
        stored_img_hsv = cv2.cvtColor(stored_img, cv2.COLOR_BGR2HSV)

        # Compute histograms
        hist_uploaded = cv2.calcHist([uploaded_img_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        hist_stored = cv2.calcHist([stored_img_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

        # Normalize histograms
        cv2.normalize(hist_uploaded, hist_uploaded, 0, 255, cv2.NORM_MINMAX)
        cv2.normalize(hist_stored, hist_stored, 0, 255, cv2.NORM_MINMAX)

        # Compare histograms using Correlation (other methods like Chi-Square can also be used)
        score = cv2.compareHist(hist_uploaded, hist_stored, cv2.HISTCMP_CORREL)

        # Keep track of the best match
        if score > best_score:
            best_score = score
            best_match = label

    return best_match, best_score

# Streamlit app
st.title("Traffic Sign Recognition")
st.image("trs.png", caption="These are the traffic signs that can be recognized by this system.", use_column_width=True)
st.write("This application allows you to upload an image of a traffic sign and identifies it by matching it against stored templates. "
         "The tool is designed to support a variety of commonly encountered traffic signs, aiding in recognition and learning.")

st.write("Upload an image of a traffic sign.")

# Load traffic sign templates
folder_path = "templates"
traffic_signs = load_traffic_signs(folder_path)

# File uploader
uploaded_file = st.file_uploader("Upload a Traffic Sign Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    uploaded_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Find the best match
    best_match, best_score = find_best_match(uploaded_img, traffic_signs)

    if best_match and best_score > 0.5:  # Score threshold for better matches
        st.success(f"Traffic Sign: {best_match}")
    else:
        st.error("No matching traffic sign found.")
