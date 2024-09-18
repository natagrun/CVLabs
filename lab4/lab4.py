import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


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


def prewitt_filter(img):
    kernel_prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernel_prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    prewitt_x = cv.filter2D(img, -1, kernel_prewitt_x)
    prewitt_y = cv.filter2D(img, -1, kernel_prewitt_y)
    return prewitt_x + prewitt_y


def sobel_filter(img):
    sobel_x = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
    sobel_y = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)
    return sobel_x + sobel_y


def scharr_filter(img):
    scharr_x = cv.Scharr(img, cv.CV_64F, 1, 0)
    scharr_y = cv.Scharr(img, cv.CV_64F, 0, 1)
    return scharr_x + scharr_y


def laplace_filter(img):
    return cv.Laplacian(img, cv.CV_64F)


def modificated_laplace_filter(img):
    kernel_modified_laplace = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    return cv.filter2D(img, -1, kernel_modified_laplace)


def display_images(original, images, titles):
    num_images = len(images)
    num_cols = (num_images + 1) // 2

    plt.figure(figsize=(15, 8))

    plt.subplot(2, num_cols, 1)
    plt.imshow(cv.cvtColor(original, cv.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    for i, (img, title) in enumerate(zip(images, titles)):
        if img.dtype != 'uint8':
            img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

        plt.subplot(2, num_cols, i + 2)
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


kernel = np.array([-0.1, 0.2, -0.1,
                   0.2, 3.0, 0.2,
                   -0.1, 0.2, -0.1])

image_cubic = cv.imread('image0.jpg')
image_cat = cv.imread('cat.jpg')

filtered_images = [linear_filter(image_cubic, kernel), linear_arithmetical_filter(image_cubic),
                   box_filter_norm_true(image_cubic), box_filter_norm_false(image_cubic),
                   gaussian_blur(image_cubic), median_filter(image_cubic), bilateral_filter(image_cubic)]
filter_titles = ['Linear', 'linear_arithmetical', 'box_filter_norm_true', 'box_filter_norm_false', 'gaussian_blur',
                 'median_filter', 'bilateral_filter']

display_images(image_cubic, filtered_images, filter_titles)

filtered_images = [prewitt_filter(image_cat), sobel_filter(image_cat),
                   scharr_filter(image_cat), laplace_filter(image_cat),
                   modificated_laplace_filter(image_cat)]
filter_titles = ['Prewitt', 'Sobel', 'Scharr', 'Laplacian', 'Modified Laplacian']

display_images(image_cat, filtered_images, filter_titles)
