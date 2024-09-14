import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('moscow.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray_image = cv2.cvtColor(image,
                          cv2.COLOR_RGB2GRAY)


def log_transform(image):
    c = 255 / np.log(1 + np.max(image))
    log_image = np.zeros_like(image, dtype=np.float64)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            log_image[i, j] = c * np.log(1 + image[i, j])
    log_image = np.array(log_image, dtype=np.uint8)

    return log_image


def gamma_correction(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        table[i] = np.clip((i / 255.0) ** inv_gamma * 255, 0, 255)
    corrected_image = cv2.LUT(image, table)

    return corrected_image


def piecewise_linear_transform(image, r1, s1, r2, s2):
    output = np.zeros_like(image, dtype=np.uint8)
    for r in range(256):
        if r < r1:
            output[image == r] = (s1 / r1) * r
        elif r1 <= r <= r2:
            output[image == r] = ((s2 - s1) / (r2 - r1)) * (r - r1) + s1
        else:
            output[image == r] = ((255 - s2) / (255 - r2)) * (r - r2) + s2

    return output


def first_window():
    log_image = log_transform(image)

    gamma_values = [0.1, 0.5, 1.2, 2.2]
    gamma_images = [gamma_correction(image, gamma) for gamma in gamma_values]

    r1, s1 = 70, 0
    r2, s2 = 140, 255
    piecewise_linear_image = piecewise_linear_transform(gray_image, r1, s1, r2, s2)

    fig, axs = plt.subplots(3, 3, figsize=(7, 7))

    axs[0, 0].imshow(image)
    axs[0, 0].set_title("Оригинальное\nизображение")
    axs[0, 0].axis('off')

    axs[0, 1].imshow(log_image)
    axs[0, 1].set_title("Логарифмическое\nпреобразование")
    axs[0, 1].axis('off')

    axs[0, 2].imshow(gamma_images[0])
    axs[0, 2].set_title("Гамма=0.1")
    axs[0, 2].axis('off')

    axs[1, 0].imshow(gamma_images[1])
    axs[1, 0].set_title("Гамма=0.5")
    axs[1, 0].axis('off')

    axs[1, 1].imshow(gamma_images[2])
    axs[1, 1].set_title("Гамма=1.2")
    axs[1, 1].axis('off')

    axs[1, 2].imshow(gamma_images[3])
    axs[1, 2].set_title("Гамма=2.2")
    axs[1, 2].axis('off')

    axs[2, 1].imshow(piecewise_linear_image, cmap='gray')
    axs[2, 1].set_title("Линейное преобразование")
    axs[2, 1].axis('off')

    axs[2, 0].axis('off')
    axs[2, 2].axis('off')

    fig.suptitle("Градационные преобразования")
    fig.canvas.manager.set_window_title("Градационные преобразования")
    fig.text(0.5, 0.02, "Закройте данное окно, чтобы перейти ко второй части задания", ha='center', fontsize=12)
    plt.show()


def image_inversion(img):
    return cv2.bitwise_not(img)


def image_and_action(img1, img2):
    return cv2.bitwise_and(img1, img2)


def image_or_action(img1, img2):
    return cv2.bitwise_or(img1, img2)


def image_xor_action(img1, img2):
    return cv2.bitwise_xor(img1, img2)


def add_image_border(ax, img, title):
    ax.imshow(img)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)


def second_window():
    # Создание первого изображения
    img1 = np.zeros((200, 400, 3), dtype="uint8")
    rectangle = np.array([[0, 0], [200, 0], [200, 200], [0, 200]])
    cv2.fillPoly(img1, [rectangle], color=(255, 255, 255))
    cv2.imwrite("drawing1.jpg", img1)

    img2 = np.zeros((200, 400, 3), dtype="uint8")
    rectangle = np.array([[150, 100], [300, 100], [300, 150], [150, 150]])
    cv2.fillPoly(img2, [rectangle], color=(255, 255, 255))
    cv2.imwrite("drawing2.jpg", img2)
    fig_main, axs_main = plt.subplots(3, 4, figsize=(9, 9))
    fig_main.canvas.manager.set_window_title("Логические операции")
    fig_main.suptitle("Логические операции")

    add_image_border(axs_main[0, 0], img1, "Изображение 1")
    add_image_border(axs_main[0, 1], image_inversion(img1), "Инверсия 1")
    add_image_border(axs_main[0, 2], img2, "Изображение 2")
    add_image_border(axs_main[0, 3], image_inversion(img2), "Инверсия 2")

    add_image_border(axs_main[1, 0], img1, "Изображение 1")
    add_image_border(axs_main[1, 1], img2, "Изображение 2")

    axs_main[1, 2].axis('off')
    axs_main[1, 3].axis('off')

    add_image_border(axs_main[2, 0], image_and_action(img1, img2), "Конъюнкция")
    add_image_border(axs_main[2, 1], image_or_action(img1, img2), "Дизъюнкция")
    add_image_border(axs_main[2, 2], image_xor_action(img1, img2), "XOR")

    axs_main[2, 3].axis('off')

    plt.tight_layout()
    plt.show()


first_window()
second_window()
