import  cv2 as cv
import  numpy as np
import matplotlib.pyplot as plt

def harris_corner_detection(image_path, thresholds):
    img = cv.imread(image_path)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    blockSize = 2
    ksize = 3
    k = 0.04
    dst = cv.cornerHarris(gray, blockSize, ksize, k)

    dst = cv.dilate(dst, np.ones(5))

    dst_resized = cv.resize(dst, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv.INTER_LINEAR)
    plt.figure(figsize=(12, 12))
    for i, threshold in enumerate(thresholds):
        img_corners = cv.resize(img.copy(), None, fx=2, fy=2,
                                interpolation=cv.INTER_LINEAR)


        img_corners[dst_resized > threshold * dst_resized.max()] = [128, 0, 128]

        plt.subplot(1, len(thresholds), i + 1)
        plt.title(f"Порог: {threshold}")
        plt.imshow(cv.cvtColor(img_corners, cv.COLOR_BGR2RGB))
        plt.axis("off")

    plt.show()


thresholds = [0.01, 0.05, 0.1]

harris_corner_detection("test_image_Smith.jpg", thresholds)

def detect_face_and_eyes(lena,eye_cascade,face_cascade,plot):
    #lena= cv.imread('lena.jpg',)
    gray = cv.cvtColor(lena,cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Шаг d: Если лица найдены
    for (x, y, w, h) in faces:
        # Нарисовать прямоугольник вокруг лица
        cv.rectangle(lena, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Шаг e: ROI для лица
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = lena[y:y + h, x:x + w]

        # Шаг f: Обнаружение глаз в ROI
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(15, 15))
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    if plot:
        # Отображение результата
        plt.figure(figsize=(10, 10))
        plt.imshow(cv.cvtColor(lena, cv.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("Обнаружение лица и глаз")
        plt.show()
    return lena

detect_face_and_eyes(cv.imread('lena.jpg'),cv.CascadeClassifier("haarcascade_eye.xml"),cv.CascadeClassifier("haarcascade_frontalface_alt2.xml"),True)

def detect_face_in_video():
    video = cv.VideoCapture("Jalinga_360.mp4")
    if video.isOpened():
        fps =video.get(cv.CAP_PROP_FPS)
        video_length = int(video.get(cv.CAP_PROP_FRAME_COUNT)) - 1
        print("Number of frames: ", video_length)
        frame_width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
        out = cv.VideoWriter('output.avi', cv.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))
        eye_cascade = cv.CascadeClassifier("haarcascade_eye.xml")
        face_cascade = cv.CascadeClassifier("haarcascade_frontalface_alt2.xml")
        frame_count =0
        while True:
            ret, frame = video.read()
            if not ret or frame_count > (video_length-1):
                break
            frame_count += 1
            out.write(detect_face_and_eyes(frame.copy(),eye_cascade,face_cascade,False))
        out.release()
    else:
        print("Video not open")
detect_face_in_video()