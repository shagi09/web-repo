import cv2
import streamlit as st
import numpy as np
import tempfile



# Load the object detection model
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'
model = cv2.dnn_DetectionModel(frozen_model, config_file)

# Load class labels
classLabels = []
file_name = 'labels.txt'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

def main():
    st.title("Object Detector")

    option = st.selectbox("Choose an option", ["Upload Image", "Upload Video", "Open Webcam"])

    if option == "Upload Image":
        upload_image()
    elif option == "Upload Video":
        upload_video()
    elif option == "Open Webcam":
        open_webcam()

def upload_image():
    file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if file is not None:
        # Read the image
        img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

        # Perform object detection on the image
        classIndex, confidence, bbox = model.detect(img, confThreshold=0.5)

        # Draw bounding boxes and labels on the image
        font_scale = 3
        font = cv2.FONT_HERSHEY_PLAIN
        for classInd, conf, boxes in zip(classIndex.flatten(), confidence.flatten(), bbox):
            cv2.rectangle(img, boxes, (255, 0, 0), 2)
            cv2.putText(img, classLabels[classInd - 1], (boxes[0] + 10, boxes[1] + 40), font, fontScale=font_scale,
                        color=(0, 255, 0), thickness=3)

        # Convert the image to RGB for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Display the image with detected objects
        st.image(img_rgb)

def upload_video():
    file = st.file_uploader("Upload Video", type=["mp4"])

    if file is not None:
        # Open the video file
        video_bytes = file.read()

        # Create a temporary file to save the video
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(video_bytes)
        temp_file.close()

        cap = cv2.VideoCapture(temp_file.name)


        while True:
            # Read a frame from the video
            ret, frame = cap.read()

            if not ret:
                break

            # Perform object detection on the frame
            classIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)

            # Draw bounding boxes and labels on the frame
            font_scale = 3
            font = cv2.FONT_HERSHEY_PLAIN
            for classInd, conf, boxes in zip(classIndex.flatten(), confidence.flatten(), bbox):
                cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                cv2.putText(frame, classLabels[classInd - 1], (boxes[0] + 10, boxes[1] + 40), font,
                            fontScale=font_scale, color=(0, 255, 0), thickness=3)

            # Convert the frame to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display the frame with detected objects
            st.image(frame_rgb)


def open_webcam():
    # Open the video file
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        if not ret:
            break

        # Perform object detection on the frame
        classIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)

        # Draw bounding boxes and labels on the frame
        font_scale = 3
        font = cv2.FONT_HERSHEY_PLAIN
        for classInd, conf, boxes in zip(classIndex.flatten(), confidence.flatten(), bbox):
            cv2.rectangle(frame, boxes, (255, 0, 0), 2)
            cv2.putText(frame, classLabels[classInd - 1], (boxes[0] + 10, boxes[1] + 40), font,
                        fontScale=font_scale, color=(0, 255, 0), thickness=3)

        # Convert the frame to RGB for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame with detected objects
        st.image(frame_rgb)

if __name__ == "__main__":
    main()
