import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def linear_filter(img, kernel):
    return cv.filter2D(img, -1, kernel)


def create_mask(size):
    mask = np.zeros((size, size), dtype=np.float32)
    row = [1] * size
    mask[0] = row
    mask[1] = row
    mask[2] = row
    mask[3] = row
    mask[4] = row
    return mask


def linear_arithmetical_filter(img):
    filter_size = 5
    max_dist = np.sqrt(8)

    mask = create_mask(5)

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



def first():
    image = cv.imread('cat.jpg')
    edges = cv.Canny(image, 50, 200)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Исходное изображение")
    plt.imshow(image[...,::-1], cmap='gray')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title("Результат Кэнни")
    plt.imshow(edges, cmap='gray')
    plt.axis('off')
    plt.show()
def process_image(image_path, apply_gaussian=True):
    # Загрузка изображения
    image = cv.imread(image_path, cv.IMREAD_COLOR)
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    filtered_images = [prewitt_filter(image), sobel_filter(image),
                       scharr_filter(image), laplace_filter(image),
                       modificated_laplace_filter(image)]
    filter_titles = ['Prewitt_lab4', 'Sobel_lab4', 'Scharr_lab4', 'Laplacian_lab4', 'Modified Laplacian_lab4']

    display_images(image, filtered_images, filter_titles)
    # Удаление шумов с помощью фильтра Гаусса (если применимо)
    if apply_gaussian:
        image_gray = cv.GaussianBlur(image_gray, (5, 5), 0)

    # Применение операторов Собеля
    sobel_x = cv.Sobel(image_gray, cv.CV_64F, 1, 0, ksize=3)  # Горизонтальный
    sobel_y = cv.Sobel(image_gray, cv.CV_64F, 0, 1, ksize=3)  # Вертикальный

    # Вычисление приближенного значения градиента
    gradient = cv.convertScaleAbs(0.5 * sobel_x + 0.5 * sobel_y)

    # Применение оператора Лапласса
    laplacian = cv.Laplacian(image_gray, cv.CV_64F)
    laplacian_abs = cv.convertScaleAbs(laplacian)

    # Применение детектора Кэнни
    canny_edges = cv.Canny(image_gray, 50, 200)

    # Отображение результатов
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.title("Исходное изображение")
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title("Оттенки серого (сглажено)" if apply_gaussian else "Оттенки серого")
    plt.imshow(image_gray, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title("Горизонтальный Собель")
    plt.imshow(cv.convertScaleAbs(sobel_x), cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title("Вертикальный Собель")
    plt.imshow(cv.convertScaleAbs(sobel_y), cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title("Приближенный градиент")
    plt.imshow(gradient, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.title("Лапласс и Кэнни")
    plt.imshow(laplacian_abs, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def preprocess_image(image_path):
    """Загрузка изображения, преобразование в оттенки серого и бинаризация."""
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV)  # Инверсия для белого фона
    return image, binary

def find_largest_contour_by_area(image_path):
    image, binary = preprocess_image(image_path)

    # Поиск контуров
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv.contourArea)

    # Отрисовка контура
    cv.drawContours(image, [largest_contour], -1, (255, 0, 0), 2)
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.title("Наибольший контур по площади")
    plt.axis('off')
    plt.show()

def find_largest_contour_by_perimeter(image_path):
    image, binary = preprocess_image(image_path)

    # Поиск контуров
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=lambda x: cv.arcLength(x, True))

    # Отрисовка контура
    cv.drawContours(image, [largest_contour], -1, (255, 0, 0), 2)
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.title("Наибольший контур по периметру")
    plt.axis('off')
    plt.show()

def approximate_contour(image_path):
    image, binary = preprocess_image(image_path)

    # Поиск контуров
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=lambda x: cv.arcLength(x, True))

    # Аппроксимация полигона
    epsilon = 0.01 * cv.arcLength(largest_contour, True)
    approx = cv.approxPolyDP(largest_contour, epsilon, True)

    # Отрисовка аппроксимированного контура
    cv.drawContours(image, [approx], -1, (255, 0, 0), 2)
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.title("Аппроксимация полигона")
    plt.axis('off')
    plt.show()

def compare_shapes(image1_path, image2_path):
    _, binary1 = preprocess_image(image1_path)
    _, binary2 = preprocess_image(image2_path)

    contours1, _ = cv.findContours(binary1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv.findContours(binary2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    match = cv.matchShapes(contours1[0], contours2[0], cv.CONTOURS_MATCH_I1, 0.0)
    print(f"Разница между контурами: {match}")


def hough_line_detection(image_path):
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    edges = cv.Canny(gray, 50, 200)

    lines = cv.HoughLines(edges, rho=1, theta=np.pi / 180, threshold=100)
    hough_image = image.copy()
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv.line(hough_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    lines_p = cv.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=10, minLineLength=50, maxLineGap=10)
    houghp_image = image.copy()
    if lines_p is not None:
        for line in lines_p:
            x1, y1, x2, y2 = line[0]
            cv.line(houghp_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv.cvtColor(edges, cv.COLOR_BGR2RGB))
    plt.title("Края (Кэнни)")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(cv.cvtColor(hough_image, cv.COLOR_BGR2RGB))
    plt.title("HoughLines")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv.cvtColor(houghp_image, cv.COLOR_BGR2RGB))
    plt.title("HoughLinesP")
    plt.axis("off")

    plt.show()




first()
process_image("cat.jpg")
process_image("cat_gauss_noise.jpg")
find_largest_contour_by_area("max_area_contour.jpg")
find_largest_contour_by_perimeter("max_area_contour.jpg")
approximate_contour("max_area_contour.jpg")
compare_shapes("goal_contour_star.jpg", "max_area_contour.jpg")
hough_line_detection("image0.jpg")
