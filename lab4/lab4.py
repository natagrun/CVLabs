import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


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


def prewwit_hui_Znaetcho(img):
    # Convert to grayscale if necessary
    if len(img.shape) == 3:  # Check if it's a color image
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    img = img.astype(np.float32) / 255.0  # Normalize and convert to float32

    kernel_prewitt_x = np.array([[-1, 0, 1],
                                 [-1, 0, 1],
                                 [-1, 0, 1]])

    kernel_prewitt_y = np.array([[-1, -1, -1],
                                 [0, 0, 0],
                                 [1, 1, 1]])

    prewitt_x = cv.filter2D(img, -1, kernel_prewitt_x)
    prewitt_y = cv.filter2D(img, -1, kernel_prewitt_y)

    # Make sure the types are correct for magnitude function
    if prewitt_x.shape == prewitt_y.shape and prewitt_x.dtype == prewitt_y.dtype:
        return cv.magnitude(prewitt_x, prewitt_y)
    else:
        raise ValueError("prewitt_x and prewitt_y must have the same size and type.")


def display_images(original, *images):
    num_images = len(images) + 1
    plt.figure(figsize=(15, 5))

    plt.subplot(1, num_images, 1)
    plt.imshow(cv.cvtColor(original, cv.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    for i, img in enumerate(images):
        plt.subplot(1, num_images, i + 2)
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        # todo добпавить как нибудь имена хз костичев не поймет (да и никто из нас тоже где чио)
        plt.title(f'Filter {i + 1}')
        plt.axis('off')

    plt.show()


kernel = np.array([-0.1, 0.2, -0.1,
                   0.2, 3.0, 0.2,
                   -0.1, 0.2, -0.1])

image_cubic = cv.imread('image0.jpg')
image_cat = cv.imread('cat.jpg')

# cv.imshow('df', image_cubic)
# cv.waitKey(0)
# cv.imshow('df', median_filter(image_cubic))
display_images(image_cubic,linear_filter(image_cubic,kernel),box_filter_norm_true(image_cubic),box_filter_norm_false(image_cubic),gaussian_blur(image_cubic),median_filter(image_cubic),bilateral_filter(image_cubic),prewwit_hui_Znaetcho(image_cat))

