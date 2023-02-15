import math
import numpy as np
import cv2
import os
from mtcnn import MTCNN
from PIL import Image
import re

from crop import crop_face_mtcnn
from crop import crop_all_folderfiles
from metrics import calc_f1_score
import getfolders


class Point2D:
    def __init__(self, coordx, coordy):
        self.x = coordx
        self.y = coordy

    def distance_to(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


# regex for picking out numbers from string
p = '[\d]+[.,\d]+|[\d]*[.][\d]+|[\d]+'

# get working directories
current = getfolders.get_current_dir()
print("current:    " + current)
traindir = getfolders.get_training_dir()  # not used
testdir = getfolders.get_testing_dir()
print("test:    " + testdir)
outputdir = getfolders.get_output_dir()
print("output:    " + outputdir)

# get video id
vidID = "10-2"

# get output paths
outmetrics = os.path.normpath(outputdir + "/output" + vidID)
print("metrics folder:    " + outmetrics)
savepath = os.path.normpath(outputdir + "/output" + vidID + ".mp4")
print("savepath:    " + savepath)

# names related to ids: GOT actors, not characters!
names = ['None', 'Isaac', 'Maisie', 'Kit', 'Emilia', 'Sophie', 'Peter', 'Lena', 'Alfie', 'Iain', 'Sean']

# create LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(current + '/trainer.yml')

# label font
font = cv2.FONT_HERSHEY_SIMPLEX


# count video frames (checking if > 100)
def count_frames(videopath):
    # Open the video file
    video = cv2.VideoCapture(videopath)

    frame_count = 0
    while True:
        ret, _ = video.read()
        if ret == False:
            break
        frame_count += 1

    # Release the video file
    video.release()

    print("Total frames in the video:", frame_count)


# calculate center of face detected rectangle
def calc_center(x, y, w, h):
    cx = int(x + w/2)
    cy = int(y + h/2)
    return Point2D(cx, cy)


# get frames from video
def get_frames(videopath):
    # Load the video
    video = cv2.VideoCapture(videopath)

    # Get the video frames
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)

    # Close the video
    video.release()

    return frames


# main workflow method for face detection, recognition, and training
def detect_faces_track(videopath, outf, videoid):
    # load the MTCNN detector
    detector = MTCNN()

    # get the video frames
    frames = get_frames(videopath)

    # initialize variables for face tracking
    face_detected = False
    face_box = None

    # initialize the tracker
    tracker = cv2.legacy.TrackerMOSSE.create()


    # loop through the frames
    i = 0
    count = 0
    # true positive
    tp = 0
    # false negative
    fn = 0
    # false positive
    fp = 0

    for frame in frames:
        i += 1
        # face_detected = False
        if i % 10 == 0:  # detect face every 10 frames
            count += 1

            if os.path.exists(outmetrics):
                cv2.imwrite(os.path.join(outmetrics, str(count) + ".jpg"), frame)

            face_detected = False
            faces_detected = False
            # detect faces in the frame
            faces = detector.detect_faces(frame)
            if len(faces) > 0:
                face_detected = True
                face_box = faces[0]['box']

        if face_detected:
            tracker.init(frame, face_box)
            # Update the tracker
            success, face_boxt = tracker.update(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if success:
                # Draw a rectangle around the face
                x, y, width, height = face_boxt
            else:
                # Draw a rectangle around the face
                x, y, width, height = face_box

            if gray.any():
                id, confidence = recognizer.predict(gray[int(y):int(y) + int(height), int(x):int(x) + int(width)])
                cv2.rectangle(frame, (int(x), int(y)), (int(x) + int(width), int(y) + int(height)), (0, 0, 255), 2)

                # Check if confidence is less them 100 ==> "0" is perfect match
                if confidence < 100:
                    id = names[id]
                    confidence = "  {0}%".format(round(100 - confidence))
                else:
                    id = "unknown"
                    confidence = "  {0}%".format(round(100 - confidence))

                cv2.putText(frame, str(id), (int(x) + 5, int(y) - 5), font, 0.7, (255, 255, 255), 1)
                cv2.putText(frame, str(confidence), (int(x) + 5, int(y) + int(height) - 5), font, 0.5, (255, 255, 0), 1)

                # metrics
                if i % 10 == 0 and os.path.exists(outmetrics):
                    data = "x: " + str(x) + "\n" + "y: " + str(y) + "\n" "width: " + str(
                        width) + "\n" + "height: " + str(height) + "\n" + str(id)
                    datapath = os.path.join(outmetrics, str(count) + ".txt")
                    manualdatapath = os.path.join(outmetrics, str(count) + "m.txt")
                    with open(datapath, "w") as file:
                        file.write(data)

                    file_exists = os.path.exists(manualdatapath)
                    if file_exists:
                        filem = open(manualdatapath, 'r')
                        mlines = filem.readlines()
                        # print(mlines)
                        if len(mlines) > 4:
                            manualx = float(re.findall(p, mlines[0])[0])
                            manualy = float(re.findall(p, mlines[1])[0])
                            manualw = float(re.findall(p, mlines[2])[0])
                            manualh = float(re.findall(p, mlines[3])[0])
                            manualid = mlines[4]
                        located = False
                        if manualx and manualy and manualw and manualh:
                            mcenter = calc_center(manualx, manualy, manualw, manualh)
                            center = calc_center(x, y, width, height)
                            print(center.distance_to(mcenter))
                            print(height)
                            if center.distance_to(mcenter) < height / 2:
                                located = True

                        if located and manualid and id == manualid:
                            tp += 1
                        elif located and manualid:
                            fp += 1


    fn = count - tp - fp

    print("true positive " + str(tp))
    print("false negative " + str(fn))
    print("false positive " + str(fp))

    f1 = calc_f1_score(tp, fp, fn)

    print("f1 score " + str(f1))

    f1datapath = outmetrics + "/f1.txt"
    with open(f1datapath, "w+") as file:
        file.write("f1 score: " + str(f1))

    # Release the window
    cv2.destroyAllWindows()

    # Re-save the video with the rectangles
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(savepath, fourcc, 30, (frame.shape[1], frame.shape[0]), True)
    for frame in frames:
        out.write(frame)
    out.release()
    count_frames(vidpath)


# LBPH training based on train dataset
def train_lbph_folder(currfolder, idf, tfaces, tids):
    for file_name in os.listdir(currfolder):
        # Get the full path of the file
        file_path = os.path.join(currfolder, file_name)

        # Check if the file is an image file
        if file_name.endswith(".jpg") or file_name.endswith(".png") or file_name.endswith(".jpeg"):
            pilimg = Image.open(file_path).convert('L')  # convert it to grayscale
            img_numpy = np.array(pilimg, 'uint8')
            tfaces.append(img_numpy)
            tids.append(idf)
    return tfaces, tids


# only mp4 files
vidpath = os.path.normpath(testdir + "/" + vidID + ".mp4")
print("vidpath:    " + vidpath)

faces = []
ids = []

for i in range(1, 11):
    train_lbph_folder(current + "/" + str(i), i, faces, ids)


#recognizer.train(faces, np.array(ids))
#recognizer.write('trainer.yml')

detect_faces_track(vidpath, outputdir, vidID)
