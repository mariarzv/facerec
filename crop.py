import math
import numpy as np
import cv2
import glob
import os
from mtcnn import MTCNN


# crop faces out of character images for faster pre-processing
def crop_face_mtcnn(imgpath, filename):
    # Load the image
    image = cv2.imread(imgpath)
    h, w = image.shape[:2]
    # Load the MTCNN model
    mtcnn = MTCNN()

    # Detect faces and facial landmarks in the image
    faces = mtcnn.detect_faces(image)

    # crop and align the face
    # adapted to create negative images
    for face in faces:
        x, y, width, height = face['box']
        x1, y1 = abs(x), abs(y)
        x2, y2 = x1 + width, y1 + height
        face_boundary = image[y1:y2, x1:x2]
        neg_boundary = image[0:y1, 0:x1]
        neg_boundary2 = image[y2:h, x2:w]

        # cv2.imwrite("00" + filename + "cropped.jpg", face_boundary)

        cv2.imwrite("01" + filename + "neg.jpg", neg_boundary)
        cv2.imwrite("02" + filename + "neg.jpg", neg_boundary2)


def crop_all_folderfiles(folderpath):
    # Iterate through all the files in the folder
    for file_name in os.listdir(folderpath):
        # Get the full path of the file
        file_path = os.path.join(folderpath, file_name)

        # Check if the file is an image file
        if file_name.endswith(".jpg") or file_name.endswith(".png") or file_name.endswith(".jpeg"):
            # Print the file name
            crop_face_mtcnn(file_path, file_name)


# crop_all_folderfiles(folderpath)