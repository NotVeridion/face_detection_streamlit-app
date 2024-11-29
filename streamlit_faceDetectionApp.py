import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64

# Webpage title and file uploader
st.title("OpenCV Face Detection")
file = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])

# Detecting faces on image 
def detectFaceOpenCVDnn(net, frame):
    # Create blob from image
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    # Set blob as input into the model
    net.setInput(blob)
    # Get detection
    detections = net.forward()
    return detections

# Image annotation
def process_detection(frame, detection, conf_thresh=0.5):
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]

    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > conf_thresh:
            x1 = int(detection[0, 0, i, 3] * frame_w)
            y1 = int(detection[0, 0, i, 4] * frame_h)
            x2 = int(detection[0, 0, i, 5] * frame_w)
            y2 = int(detection[0, 0, i, 6] * frame_h)

        # Save bounding box of face detected into bboxes
        line_thickness = max(1, int(round(frame_h / 200)))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), line_thickness, cv2.LINE_8)

    return frame

# Loading DNN model
@st.cache_resource
def load_model():
    modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

    return net

# Generating download link
def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href

net = load_model()

# Check for uploaded image
if file is not None:
    # Use Image from PIL to read in image file (saved in RGB order)
    image = np.array(Image.open(file))

    placeholders = st.columns(2)

    placeholders[0].image(image)
    placeholders[0].text("Original Image")

    conf_thresh = st.slider("SET Confidence Threshold", min_value=0.0, max_value=1.0, step=0.01, value=0.5)

    # Get detections using the model
    detections = detectFaceOpenCVDnn(net, image)

    # Process image with annotations and filter by confidence threshold
    out_image = process_detection(image, detections, conf_thresh)

    placeholders[1].image(out_image)
    placeholders[1].text("Image with Face Detection")

    # Converting image from numpy array to PIL
    out_image = Image.fromarray(out_image)

    # Creating link for download
    st.markdown(get_image_download_link(out_image, "face_output.jpg", "Download output file"), unsafe_allow_html=True)




