import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def imageX2(im):
    new_size = np.array(im.shape[:2]) * 2
    resize_image = cv.resize(im, new_size)
    cv.imshow('', resize_image)
    cv.waitKey(0)

def rotate_Image():
    rotate_matrix=cv.getRotationMatrix2D(np.array(image.shape[:2])/2,45,1)
    rotate_image = cv.warpAffine(image,rotate_matrix,np.array(image.shape[:2]))
    imageX2(rotate_image)

def resize_Image():
    new_size = np.array([image.shape[0]//2, image.shape[1]])
    resize_image1 = cv.resize(image, new_size)
    cv.imshow('image resize with function resize', resize_image1)
    cv.waitKey(0)
    M_scale = np.array([[0.5, 0, 0],
                        [0, 1, 0]])
    # Выполняем аффинное преобразование
    scaled_image = cv.warpAffine(image, M_scale, new_size)
    # Отображаем изображение
    cv.imshow('Scaled Image - Matrix', scaled_image)
    cv.waitKey(0)

def offset_Image():
    m_offset = np.float32([[1, 0, 50],
                        [0, 1, 50]])

    size = np.array([image.shape[0], image.shape[1]])
    offset_image = cv.warpAffine(image, m_offset,size)
    imageX2(offset_image)

def XYreflection_Image():
    (h,w) = image.shape[:2]
    x_reflection = np.float32([[1, 0, 0], [0, -1, h]])
    y_reflection =  np.float32([[-1, 0, w], [0, 1, 0]])
    x_image = cv.warpAffine(image, x_reflection,(w,h))
    y_image = cv.warpAffine(image, y_reflection,(w,h))
    cv.imshow('X reflection', x_image)
    cv.waitKey(0)
    cv.imshow('Y reflection', y_image)
    cv.waitKey(0)

def shear_image():
    M = np.float32([[1, 0.2, 0], [0.2, 1, 0]])
    sheared_image = cv.warpAffine(image, M, (int(image.shape[1] * 1.5), int(image.shape[0] * 1.5)))
    cv.imshow('Y reflection', sheared_image)
    cv.waitKey(0)


def homography_books():
    book2 = cv.imread('book2.jpg')
    book2_pts = np.array([[141, 131], [480, 159], [493, 630], [64, 601]])
    book1 = cv.imread('book1.jpg')
    book1_pts = np.array([[318, 256], [534, 372], [316, 670], [73, 473]])
    h, status = cv.findHomography(book2_pts, book1_pts)
    im_out = cv.warpPerspective(book2, h, (book1.shape[1], book1.shape[0]))
    cv.imshow("Source Image", book2)
    cv.waitKey(0)
    cv.imshow("Destination Image", book1)
    cv.waitKey(0)
    cv.imshow("Warped Source Image", im_out)
    cv.waitKey(0)
    dst_pts = np.array([[0, 0], [300, 0], [300, 400], [0, 400]], dtype=np.float32)
    H, _ = cv.findHomography(book1_pts, dst_pts)
    cover = cv.warpPerspective(book1, H, (300, 400))
    cv.imshow("Cover", cover)
    cv.waitKey(0)
    
def homography_square():
    times_square = cv.imread('times-square.jpg')
    new_ad = cv.imread('first-image.jpg')
    src_pts = np.array([[895, 450], [1075, 340], [1005, 200], [870, 340]], dtype=np.float32)
    src_pts= src_pts[::-1]
    height, width, _ = new_ad.shape

    dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)

    H, _ = cv.findHomography(dst_pts, src_pts)

    warped_ad = cv.warpPerspective(new_ad, H, (times_square.shape[1], times_square.shape[0]))
    mask = np.zeros_like(times_square, dtype=np.uint8)

    cv.fillConvexPoly(mask, np.int32(src_pts), (255, 255, 255))
    mask_inv = cv.bitwise_not(mask)
    times_square_bg = cv.bitwise_and(times_square, mask_inv)
    result = cv.add(times_square_bg, warped_ad)
    cv.imshow("Result", result)
    cv.waitKey(0)

image= cv.imread("cat.jpg")
rotate_Image()
resize_Image()
offset_Image()
XYreflection_Image()
shear_image()
homography_books()
homography_square()

