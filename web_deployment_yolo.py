import streamlit as st
from PIL import Image
import zipfile
import cv2
import numpy as np
import os

extract_path = os.getcwd()
os.path.join(extract_path, 'best.onnx')
with zipfile.ZipFile('best.zip', 'r') as zip_ref:
    zip_ref.extractall(extract_path)

labels = ['person','car','chair','bottle','pottedplant','bird','dog','sofa','bicycle','horse','boat','motorbike','cat',
          'tvmonitor','cow','sheep','aeroplane','train','diningtable','bus']

# Loading YOLO Model
yolo = cv2.dnn.readNetFromONNX('best.onnx')
yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def predictions(image):
    row, col, d = image.shape
    # converting image into square image
    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image
    # getting prediction
    input_wh = 640
    blob = cv2.dnn.blobFromImage(input_image, 1 / 255, (input_wh, input_wh), swapRB=True, crop=False)
    yolo.setInput(blob)
    preds = yolo.forward()
    # Filtering using confidence (0.4) and probability score (.25)
    detections = preds[0]
    boxes = []
    confidences = []
    classes = []

    image_w, image_h = input_image.shape[0:2]
    x_factor = image_w / input_wh
    y_factor = image_h / input_wh

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4]  # confidence of detecting an object
        if confidence >= 0.4:
            class_score = row[5:].max()  # maximum probability object
            class_id = row[5:].argmax()
            if class_score >= 0.25:
                cx, cy, w, h = row[0:4]  # getting the centre_x,centre_y,w,h of bounding box
                # constructing the bounding box
                left = int((cx - 0.5 * w) * x_factor)
                top = int((cy - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                box = np.array([left, top, width, height])

                confidences.append(confidence)
                boxes.append(box)
                classes.append(class_id)

    boxes = np.array(boxes).tolist()
    confidences = np.array(confidences).tolist()

    # Non Maximum Suppression
    index = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45).flatten()

    # Drawing the boxes
    for ind in index:
        x, y, w, h = boxes[ind]
        bb_conf = np.round(confidences[ind], 2)
        classes_id = classes[ind]
        class_name = labels[classes_id]
        colours = generate_colours(classes_id)
        text = f'{class_name.upper()}: {bb_conf}'
        cv2.rectangle(image, (x, y), (x + w, y + h), colours, 2)
        cv2.rectangle(image, (x, y - 30), (x + w, y), colours, -1)
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0), 1)

    return image


def generate_colours(ID):
    color_map = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 0),    # Maroon
        (0, 128, 0),    # Olive
        (0, 0, 128),    # Navy
        (128, 128, 0),  # Olive Green
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
        (255, 165, 0),  # Orange
        (128, 128, 128),  # Gray
        (128, 255, 0),  # Lime
        (255, 128, 0),  # Dark Orange
        (128, 0, 255),  # Violet
        (255, 128, 128),  # Light Pink
        (255, 255, 128),  # Pale Yellow
        (173, 216, 230),  # Light Blue
    ]

    # Using the color map based on the class ID
    return color_map[ID]

def main():
    st.title('🚀 Welcome to my Multiple Object Detector Web App!')
    st.write(
        "Hello! I am Mayank Chandak, a student at IIT Madras with a passion for Artificial Intelligence and Machine Learning. "
        "This web app utilizes a model trained on the VOC2012 dataset using YOLOv5 and can predict 20 different objects in images. "
        "Upload an image, and the model will detect and annotate objects in the scene. The objects that can be detected are: "
        "person, car, chair, bottle, potted plant, bird, dog, sofa, bicycle, horse, boat, motorbike, cat, "
        "TV/monitor, cow, sheep, aeroplane, train, dining table, bus.")

    st.sidebar.header("Details")

    # File upload
    upload = st.file_uploader(label="Upload Image Here:", type=["png", "jpg", "jpeg"])

    if upload:
        # Sidebar customization
        st.sidebar.subheader("File Details:")
        st.sidebar.text(f"File Name: {upload.name}")
        st.sidebar.text(f"File Type: {upload.type}")

        file_extension = upload.name.split(".")[-1].lower()

        if file_extension in ["png", "jpg", "jpeg"]:
            # For images
            img = Image.open(upload)
            st.image(img, caption="Uploaded Image", use_column_width=True)

            # Convert to OpenCV format (only if needed, remove if your model works with RGB)
            image_cv = np.array(img)

            # Make predictions and get annotated image
            annotated_image = predictions(image_cv)

            # Display the annotated image
            st.image(annotated_image, caption="Annotated Image", use_column_width=True)

        else:
            st.warning("Unsupported file format. Please upload an image (png, jpg, jpeg) or a video (mp4).")

    st.sidebar.subheader("Citation Information:")
    st.sidebar.markdown(
        "This model was trained on the [PASCAL Visual Object Classes Challenge 2012 (VOC2012)](http://www.pascal-network.org/challenges/VOC/voc2012/workshop/index.html) dataset. "
        "@misc{pascal-voc-2012,\n"
        "    author = \"Everingham, M. and Van~Gool, L. and Williams, C. K. I. and Winn, J. and Zisserman, A.\",\n"
        "    title = \"The {PASCAL} {V}isual {O}bject {C}lasses {C}hallenge 2012 {(VOC2012)} {R}esults\",\n"
        "    howpublished = \"http://www.pascal-network.org/challenges/VOC/voc2012/workshop/index.html\""
        ")")
if __name__ == "__main__":
    main()
