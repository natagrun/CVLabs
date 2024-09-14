import cv2 as cv
import numpy as np


def linear_filter(img, kernel):
    return cv.filter2D(img, -1, kernel)


def input_mask(size):
    mask = np.zeros((size, size), dtype=np.float32)
    print("Введите значения маски {}x{} по строкам (через пробел):".format(size, size))
    for i in range(size):
        row = input("Строка {}: ".format(i + 1)).strip().split()
        mask[i] = [float(val) for val in row]
    return mask


def linear_arithmetical_filter(img):
    filter_size = 5
    max_dist = np.sqrt(8)

    mask = input_mask(5)

    for i in range(filter_size):
        for j in range(filter_size):
            dist = np.sqrt((i - 2) ** 2 + (j - 2) ** 2)
            mask[i, j] = (max_dist - dist) / max_dist

    mask = mask / np.sum(mask)

    return cv.filter2D(img, -1, mask)


def box_filter_norm_true(img):
    return cv.boxFilter(img, ddepth=-1, ksize=(5, 5), normalize=True)


def box_filter_norm_false(img):
    return cv.boxFilter(img, ddepth=-1, ksize=(5, 5), normalize=False)


def gaussian_blur(img):
    return cv.GaussianBlur(img, ksize=(5, 5), sigmaX=0, sigmaY=0)


def median_filter(img):
    return cv.medianBlur(img, 5)


def bilateral_filter(img):
    #                           ERRRGH wHAta SigMa????!!
    return cv.bilateralFilter(img, -1, sigmaColor=10, sigmaSpace=5)


kernel = np.array([-0.1, 0.2, -0.1,
                   0.2, 3.0, 0.2,
                   -0.1, 0.2, -0.1])

image = cv.imread('image0.jpg')

cv.imshow('df', image)
cv.waitKey(0)
cv.imshow('df', median_filter(image))

