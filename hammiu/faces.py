from imutils import paths
import numpy as np
import cv2
import os

# -----------------------------
#   FUNCTIONS: apply face detector to a image and return coordinates of bb
# -----------------------------
def detect_faces(net, image, minConfidence=0.5):    # phát hiện face mà có Probability < minConfidence thì bỏ qua
    # Lấy thông số ảnh
    (h, w) = image.shape[:2]
    # chuyển ảnh thành blob (bước tiền xử lý trước khi đưa vào model)
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    # đưa blob vào network để nhận được đầu ra (các detections)
    # khởi tạo list để chứa các bounding boxes dự đoán
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    # Duyệt qua các detections
    for i in range(0, detections.shape[2]):
        # Lấy confidence (probability) liên quan đến detection
        confidence = detections[0, 0, i, 2]
        # Lọc bớt các detection có confidence thấp hơn so với minConfidence
        if confidence > minConfidence:
            # Tính lại tọa độ của bounding box, chứng ta SSD trả vê (xmin, ymin, xmax, ymax) tương đối so với ảnh ban đầu.
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # update list of boxes
            boxes.append((startX, startY, endX, endY))
    # Return boxes
    return boxes

""" 
    1. Duyệt qua tất cả ảnh trong caltech_images, 
    2. đếm số ảnh cho mỗi người 
    3. loại bỏ ai mà số ảnh khuôn mặt ít hơn N (tránh imbalance)
    4. áp detect_faces lên đó
    5. Trích xuất Face ROI
    6. Trả về face ROIs và class label (tên của người) 
"""
def load_face_dataset(inputPath, net, minConfidence=0.5, minSamples=15):
    # Lấy đường dẫn các ảnh, tách tất cả các tên lại, đếm xem có bao nhiêu tên duy nhất và mỗi loại bao nhiêu
    imagePaths = list(paths.list_images(inputPath))
    names = [p.split(os.path.sep)[-2] for p in imagePaths]      # trả về list of names (có trùng nhau)
    (names, counts) = np.unique(names, return_counts=True)      # trả về 2 numpy array, 1 cái là tên, 1 cái là số lượng tên tương ứng
    names = names.tolist()      # chuyển về dạng list
    # Khởi tạo list để lưu các extracted faces and labels tương ứng
    faces = []
    labels = []
    # Duyệt qua các image paths
    for imagePath in imagePaths:
        # Load image từ disk và trích xuất tên tương ứng
        image = cv2.imread(imagePath)
        name = imagePath.split(os.path.sep)[-2]
        # Chỉ xử lý khuôn mặt của người có số lượng lớn hơn threshold = minSamples
        if counts[names.index(name)] < minSamples:      # names.index(name) lấy chỉ số trong list có value là name
            continue    # tiếp tục lặp, ko xử lý bên dưới
        # Thực hiện face detection, nhận lại được list các boxes với tuples là (xmin, ymin, xmax, ymax)
        boxes = detect_faces(net=net, image=image, minConfidence=minConfidence)
        # Duyệt qua các boxes tìm được trong ảnh
        for (startX, startY, endX, endY) in boxes:
            # Trích xuất FACE ROI và chuyển về grayscale image
            faceROI = image[startY:endY, startX:endX]
            faceROI = cv2.resize(faceROI, (47, 62))     # resize về size cố định
            faceROI = cv2.cvtColor(faceROI, cv2.COLOR_BGR2GRAY)
            # Update face và label lists bên trên 
            faces.append(faceROI)
            labels.append(name)
    # chuyển face và label lists về numpy array
    faces = np.array(faces)
    labels = np.array(labels)
    # Return a 2-tuple of the faces and labels
    return faces, labels


