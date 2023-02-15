import numpy as np
import cv2

# alternative methods tried before, not used in current pipeline

# def train_dnn(imagefolder):
#     # load the DNN face detector
#     detector = cv2.dnn.readNetFromCaffe(protofilder, caffeemodelfolder)
#
#     # load the face recognition model
#     model = cv2.dnn.readNetFromTorch("path/to/face_recognition_model.t7")
#
#     # create an empty list to store the images
#     images = []
#
#     # create an empty list to store the labels
#     labels = []
#
#     # loop through all the images in the given folder
#     for image_path in glob.glob(imagefolder):
#         # Read the image
#         image = cv2.imread(image_path)
#         # Get the image height and width
#         h, w = image.shape[:2]
#         # Add the face to the images list
#         images.append(image)
#         # Add the label to the labels list
#         labels.append("maisie")
#
#     # create the face recognition XML file
#     model.write("face_recognition.xml", images, labels)

# def detect_face_frame_lbp():
#     # Load the cascade classifier
#     face_cascade = cv2.CascadeClassifier(cascadefrontalpath)
#
#     # Read the input image
#     img = cv2.imread(imagepath)
#
#     # Convert the image to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     # Detect faces in the image
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
#
#     # Draw rectangles around the detected faces
#     for (x, y, w, h) in faces:
#         cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#
#     # Show the output image
#     cv2.imshow('img', img)
#     cv2.waitKey()
#     cv2.destroyAllWindows()

# def train_lbp(posfolder, negfolder):
#     # Create the Local Binary Patterns Histograms (LBPH) face recognizer
#     face_recognizer = cv2.face.LBPHFaceRecognizer_create()
#
#     # Get the directory containing the training images
#     training_data_dir = posfolder
#
#     # Get the images and labels
#     images = []
#     labels = []
#     for image_path in os.listdir(training_data_dir):
#         image = cv2.imread(os.path.join(training_data_dir, image_path), cv2.IMREAD_GRAYSCALE)
#         images.append(image)
#         labels.append(2)
#
#
#     # Train the classifier
#     face_recognizer.train(images, np.array(labels))
#
#     face_recognizer.save("classifier.xml")

# def classify_face(imagepath):
#     # Load the pre-trained classifier
#     face_cascade = cv2.CascadeClassifier(cascadeclassifierpath)
#
#     # Read an image
#     img = cv2.imread(imagepath)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     # Detect faces in the image
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
#
#     # Draw rectangles around the faces
#     for (x, y, w, h) in faces:
#         cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#
#     # Show the image
#     cv2.imshow("Face Detection", img)
#     cv2.waitKey(0)

#train_dnn(testfolder)
#train_lbp(testfolder, negfolder)
#classify_face(imgpath)