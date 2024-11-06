import cv2
import numpy as np

#1
img = cv2.imread('type_2G.jpg', cv2.IMREAD_REDUCED_GRAYSCALE_2)
cv2.imshow('Исходное изображение', img)
cv2.waitKey(0)
# cv2.destroyAllWindows()
kernel = np.ones((5,5), np.uint8)
erosion = cv2.erode(img, kernel, iterations=1)
cv2.imshow('Изображение после однократного примененияя эрозии', erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()

#2
img = cv2.imread('black_sq.jpg', cv2.IMREAD_REDUCED_GRAYSCALE_2)
cv2.imshow('Исходное изображение', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
kernel = np.ones((5,5), np.uint8)
erosion = cv2.erode(img, kernel, iterations=1)
cv2.imshow('Изображение после эрозии (квадраты по 5 пикселей)', erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()

#3
img = cv2.imread('type_2G.jpg', cv2.IMREAD_REDUCED_GRAYSCALE_2)
cv2.imshow('Исходное изображение', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
kernel = np.ones((5,5), np.uint8)
dilation = cv2.dilate(img, kernel, iterations=1)
cv2.imshow('Изображение после дилатации (ядро 5 на 5)', dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()

#4
img = cv2.imread('black_sq.jpg', cv2.IMREAD_REDUCED_GRAYSCALE_2)
kernel = np.ones((5,5), np.uint8)
dilated = cv2.dilate(img, kernel, iterations=1)
eroded = cv2.erode(dilated, kernel, iterations=1)
cv2.imshow('Исходное изображение', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Изображение после применения дилатации и эрозии', eroded)
cv2.waitKey(0)
cv2.destroyAllWindows()

#5
img = cv2.imread('black_sq.jpg', cv2.IMREAD_REDUCED_GRAYSCALE_2)
kernel_square = np.ones((5,5), np.uint8)
kernel_cross = np.array([[0, 0, 0, 1, 0],
                         [0, 0, 0, 1, 0],
                         [1, 1, 1, 1, 1],
                         [0, 0, 0, 1, 0],
                         [0, 0, 0, 1, 0]], np.uint8)
kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
dilate_square = cv2.dilate(img, kernel_square, iterations = 1)
dilate_cross = cv2.dilate(img, kernel_cross, iterations = 1)
dilate_ellipse = cv2.dilate(img, kernel_ellipse, iterations = 1)
cv2.imshow('Исходное изображение', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Дилатация вида: квадрат', dilate_square)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Дилатация вида: крест', dilate_cross)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Дилатация вида: матрица из задания', dilate_ellipse)
cv2.waitKey(0)
cv2.destroyAllWindows()

#6
img = cv2.imread('type_j_open.jpg',cv2.IMREAD_REDUCED_GRAYSCALE_2)
kernel = np.zeros((5,5), np.uint8)
morphological_gradient = cv2.morphologyEx(img, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_RECT,(13,13)))
cv2.imshow('Исходное изображение', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Изображение после размыкания (удаление шума соль)', morphological_gradient)
cv2.waitKey(0)
cv2.destroyAllWindows()

#7
img = cv2.imread('type_j_close.jpg', cv2.IMREAD_REDUCED_GRAYSCALE_2)
kernel = np.ones((13,13), np.uint8)
morphological_gradient = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
cv2.imshow('Исходное изображение', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Изображение после замыкания (удаление шума перец)', morphological_gradient)
cv2.waitKey(0)
cv2.destroyAllWindows()

#8
img = cv2.imread('type_j.jpg', cv2.IMREAD_REDUCED_GRAYSCALE_2)
kernel = np.ones((5,5), np.uint8)
morphological_gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
cv2.imshow('Исходное изображение', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Изображение после использования морфологического градиента', morphological_gradient)
cv2.waitKey(0)
cv2.destroyAllWindows()


#9
img = cv2.imread('type_j.jpg', cv2.IMREAD_REDUCED_GRAYSCALE_2)
kernel = np.ones((21,21), np.uint8)
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
cv2.imshow('Исходное изображение', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Изображение после применения TopHat', tophat)
cv2.waitKey(0)
cv2.destroyAllWindows()

noise = tophat - img
cv2.imshow('Шум', noise)
cv2.waitKey(0)
cv2.destroyAllWindows()

#10
img = cv2.imread('type_j.jpg', cv2.IMREAD_REDUCED_GRAYSCALE_2)
kernel = np.ones((23,23), np.uint8)
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
cv2.imshow('Исходное изображение', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Изображение после применения BlackHat', blackhat)
cv2.waitKey(0)
cv2.destroyAllWindows()

