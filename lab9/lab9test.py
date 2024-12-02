import cv2
import matplotlib.pyplot as plt
import numpy as np


def first():
    src = cv2.imread('threshold.jpg', cv2.IMREAD_GRAYSCALE)

    _, binary_32 = cv2.threshold(src, 32, 255, cv2.THRESH_BINARY)
    _, binary_128 = cv2.threshold(src, 128, 255, cv2.THRESH_BINARY)
    _, otsu_binary = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    images = [src, binary_32, binary_128, otsu_binary]
    titles = ['Original', 'Binary Thresh = 32', 'Binary Thresh = 128', 'Otsu Threshold']

    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def second():
    src = cv2.imread('adapt_threshold.jpg', cv2.IMREAD_GRAYSCALE)

    adaptive_mean = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY, 11, 2)
    adaptive_gaussian = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, 11, 2)

    _, otsu_binary = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    images = [src, adaptive_mean, adaptive_gaussian, otsu_binary]
    titles = ['Original', 'Adaptive Mean', 'Adaptive Gaussian', 'Otsu Threshold']

    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def region_growing(image, seed_points, threshold=10):
    rows, cols = image.shape
    result = np.array(image)  # Результирующее изображение

    # 8-связная смежность
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for seed in seed_points:
        x, y = seed
        intensity = image[y, x]  # Интенсивность начальной точки
        visited = np.zeros_like(image, dtype=bool)  # Маска посещённых пикселей
        queue = [(x, y)]  # Очередь для обработки
        visited[y, x] = True

        while queue:
            cx, cy = queue.pop(0)
            result[cy, cx] = 255  # Закрашиваем пиксель белым цветом

            for dx, dy in neighbors:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < cols and 0 <= ny < rows and not visited[ny, nx]:
                    if abs(int(image[ny, nx]) - int(intensity)) <= threshold:
                        queue.append((nx, ny))
                        visited[ny, nx] = True
    return result

# Загрузка изображен
def third():
    input_image = cv2.imread('region_growing.jpg', cv2.IMREAD_GRAYSCALE)
    if input_image is None:
        raise FileNotFoundError("Файл region_growing.jpg не найден.")

    # Начальные точки (координаты заданы в формате (x, y))
    seed_points = [(176, 255), (229, 405), (347, 165)]

    # Применение метода выращивания областей
    result_image = region_growing(input_image, seed_points, threshold=0)

    # Отображение оригинального и результата рядом с использованием matplotlib
    plt.figure(figsize=(10, 5))

    # Оригинальное изображение
    plt.subplot(1, 2, 1)
    plt.imshow(input_image, cmap='gray')
    plt.title('Оригинальное изображение')
    plt.axis('off')

    # Результирующее изображение
    plt.subplot(1, 2, 2)
    plt.imshow(result_image, cmap='gray')
    plt.title('Выращенные области')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


first()
second()
third()
