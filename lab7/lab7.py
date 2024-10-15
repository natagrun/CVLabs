import matplotlib.pyplot as plt
import numpy as np
from skimage import io


def butterworth_lowpass_filter(shape, D0, n):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    x = np.linspace(-ccol, ccol - 1, cols)
    y = np.linspace(-crow, crow - 1, rows)
    X, Y = np.meshgrid(x, y)
    D = np.sqrt(X ** 2 + Y ** 2)
    h = 1 / (1 + (D / D0) ** (2 * n))
    return h


def first_task():
    image = io.imread('white_sq.jpg', as_gray=True)

    f = np.fft.fft2(image)
    f_shifted = np.fft.fftshift(f)

    d0_values = [0.05, 0.5, 10.0]
    n = 2

    fig, axes = plt.subplots(len(d0_values), 3, figsize=(10, 8))

    for i, D0 in enumerate(d0_values):
        h = butterworth_lowpass_filter(image.shape, D0, n)

        g = f_shifted * h

        g_shifted = np.fft.ifftshift(g)
        image_filtered = np.fft.ifft2(g_shifted)
        image_filtered = np.abs(image_filtered)

        axes[i, 0].imshow(image, cmap='gray')
        axes[i, 0].set_title('Исходное изображение')
        axes[i, 1].imshow(np.log(np.abs(f_shifted) + 1), cmap='gray')
        axes[i, 1].set_title('Амплитудный спектр')
        axes[i, 2].imshow(image_filtered, cmap='gray')
        axes[i, 2].set_title(f'Фильтрованное изображение D0={D0}')

    plt.tight_layout()
    plt.show()


def second_task():
    image_moscow = io.imread('moscow.jpg', as_gray=True)
    f_moscow = np.fft.fft2(image_moscow)
    f_moscow_shifted = np.fft.fftshift(f_moscow)

    amplitude_spectrum = np.log(np.abs(f_moscow_shifted) + 1)

    f_moscow_shifted = np.fft.ifftshift(f_moscow_shifted)
    image_reconstructed = np.fft.ifft2(f_moscow_shifted)
    image_reconstructed = np.abs(image_reconstructed)

    # Визуализация
    plt.figure(figsize=(15, 5))

    # Исходное изображение
    plt.subplot(1, 3, 1)
    plt.imshow(image_moscow, cmap='gray')
    plt.title('Исходное изображение')

    plt.subplot(1, 3, 2)
    plt.imshow(amplitude_spectrum, cmap='gray')
    plt.title('Амплитудный спектр')

    plt.subplot(1, 3, 3)
    plt.imshow(image_reconstructed, cmap='gray')
    plt.title('Реконструированное изображение')

    plt.tight_layout()
    plt.show()


first_task();
second_task();
