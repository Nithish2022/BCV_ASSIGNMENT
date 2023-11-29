import streamlit as st
import cv2
from PIL import Image
import numpy as np

# Title and description for the app
st.title('Basics of Computer Vision Assignment')
st.write('Upload an image for edge and corner detection')

# Function for different filtering techniques
def filter_image(image, filter_type):
    if filter_type == 'Gaussian Blur':
        return cv2.GaussianBlur(image, (5, 5), 0)
    elif filter_type == 'Median Blur':
        return cv2.medianBlur(image, 5)
    elif filter_type == 'Bilateral Filter':
        return cv2.bilateralFilter(image, 9, 75, 75)
    else:
        return image

# Function for different edge detection techniques
def detect_edges_sobel(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    edges = np.sqrt(sobelx**2 + sobely**2)
    edges = np.uint8(edges)
    return edges

def detect_edges_canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

def detect_edges_log(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Laplacian(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_64F)
    edges = np.uint8(np.absolute(edges))
    return edges

def detect_edges_dog(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gaussian1 = cv2.GaussianBlur(gray, (5, 5), 0)
    gaussian2 = cv2.GaussianBlur(gray, (9, 9), 0)
    dog = cv2.absdiff(gaussian1, gaussian2)
    return dog

# Function for different corner detection techniques
def detect_corners_harris(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    corners = cv2.dilate(corners, None)
    image[corners > 0.01 * corners.max()] = [0, 0, 255]
    return image

def detect_corners_sift(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints = sift.detect(gray, None)
    image = cv2.drawKeypoints(image, keypoints, None)
    return image

# File uploader for the image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    image = np.array(Image.open(uploaded_file).convert('RGB'))
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Perform filtering, edge, and corner detection on the image
    st.subheader('Filtering')
    filter_type_edge = st.selectbox('Select Edge Filtering Method:',
                                    ('None', 'Gaussian Blur', 'Median Blur', 'Bilateral Filter'))

    st.subheader('Edge Detection')
    edge_option = st.radio('Select Edge Detection Method:',
                           ('Sobel', 'Canny', 'LoG', 'DoG'))

    st.subheader('Corner Detection')
    corner_option = st.radio('Select Corner Detection Method:',
                             ('Harris Corner', 'SIFT'))

    if st.button('Filter, Detect Edges, and Corners'):
        filtered_image = image.copy()
        if filter_type_edge != 'None':
            filtered_image = filter_image(filtered_image, filter_type_edge)

        if edge_option == 'Sobel':
            edge_image = detect_edges_sobel(filtered_image)
            st.image(edge_image, caption=f'Edges Detected (Sobel after {filter_type_edge})', use_column_width=True)
        elif edge_option == 'Canny':
            edge_image = detect_edges_canny(filtered_image)
            st.image(edge_image, caption=f'Edges Detected (Canny after {filter_type_edge})', use_column_width=True)
        elif edge_option == 'LoG':
            edge_image = detect_edges_log(filtered_image)
            st.image(edge_image, caption=f'Edges Detected (LoG after {filter_type_edge})', use_column_width=True)
        elif edge_option == 'DoG':
            edge_image = detect_edges_dog(filtered_image)
            st.image(edge_image, caption=f'Edges Detected (DoG after {filter_type_edge})', use_column_width=True)

        if corner_option == 'Harris Corner':
            corner_image = detect_corners_harris(filtered_image)
            st.image(corner_image, caption=f'Corners Detected (Harris Corner)', use_column_width=True)
        elif corner_option == 'SIFT':
            corner_image = detect_corners_sift(filtered_image)
            st.image(corner_image, caption=f'Corners Detected (SIFT)', use_column_width=True)
