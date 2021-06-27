# USAGE
# python face_recognition_LBPs.py --input caltech_faces

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from hammiu.faces import load_face_dataset, detect_faces
import numpy as np
import argparse
import time
import imutils
import os
import cv2


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True, help="path to the input directory of images")     # images folder
ap.add_argument("-f", "--face", type=str, default="face_detector", help="path to the face detector model directory")    # DL face detector
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load serialized face detector from disk
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the CALTECH faces datasets
print("[INFO] loading dataset...")
(faces, labels) = load_face_dataset(args["input"], net, minConfidence=0.5, minSamples=20)   # phần này là duyệt qua các ảnh, trích xuất faces và lấy labels tương ứng
print("[INFO] {} images in dataset".format(len(faces)))

# encode string labels as integers (cần cho LBPs làm việc)
le = LabelEncoder()
labels = le.fit_transform(labels)
print("[INFO] labels".format(labels))

# phân chia training và testing dataset
(train_x, test_x, train_y, test_y) = train_test_split(faces, labels, test_size=0.25, stratify=labels, random_state=42)

# traing LBPs face recognizer
print("[INFO training face recognizer]")    # grid_x, grid_y (fine tune) kiểm soát số grif cells trong, ở đây dùng 8x8 thay vì 7x7 như bài báo gốc, acc tăng nhưng time, memory cũng tăng
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=16, grid_x=8, grid_y=8)     # raadius, neighbors là parameters của LBP image descriptor
# xem training process laau ko
start = time.time()
recognizer.train(train_x, train_y)
end = time.time()
print("[INFO] training took {:.4f} seconds".format(end-start))

# khởi tạo list of predictions và confidence scores
print("[INFO] gathering predictions...")
predictions = []
confidence = []
start = time.time()

# Duyệt qua các test data
for i in range(0, len(test_x)):
    # phân loại khuôn mặt và update 2 lists bên trên
    (prediction, conf) = recognizer.predict(test_x[i])      # prediction laf integer label of subject, conf - là khi distance của test image với closet image trong training dataset
    predictions.append(prediction)
    confidence.append(conf)

# xem predict mất nhiều time ko
end = time.time()
print("[INFO] inference took {:.4f} seconds".format(end-start))

# show classifification_report
print(classification_report(test_y, predictions, target_names=le.classes_))

# Bước cuối cùng là hiển thị kết quả
# laays ngẫu nhiên các testing data
idxs = np.random.choice(range(len(test_y)), size=10, replace=False)

# duyệt qua các testing data đã chọn để hiển thị
for i in idxs:
    # lấy name thật sự và name dự đoán
    predName = le.inverse_transform([predictions[i]])   # chuyển từ index về string label
    actualName = le.classes_[test_y[i]]

    # lấy ảnh khuôn mặt, resize để dễ hiển thị
    face = np.dstack([test_x[i]] * 3)
    face = imutils.resize(face, width=250)

    # ghi tên dự đoán và tên thật lên image
    cv2.putText(face, "pred: {}".format(predName), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(face, "actual: {}".format(actualName), (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)    

    print("[INFO] prediction : {}, actual: {}, confidence: {:.2f}".format(predName, actualName, confidence[i]))

    cv2.imshow("Face", face)
    cv2.waitKey(0)  # nhấn phím bất kì để chuyển qua ảnh khác
