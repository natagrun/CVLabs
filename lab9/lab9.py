import cv2
from matplotlib import pyplot as plt

threshold_image = 'threshold.jpg'
adapt_threshold_image = 'adapt_threshold.jpg'


def first():
    src = cv2.imread(threshold_image, cv2.IMREAD_GRAYSCALE)

    if src is None:
        print(f"Не удалось загрузить изображение по пути: {threshold_image}")
        exit()

    maxval = 255
    thresh_values = [32, 128]

    results = {}

    for thresh in thresh_values:
        retval, dst = cv2.threshold(src, thresh, maxval, cv2.THRESH_BINARY)
        results[f'THRESH_BINARY_{thresh}'] = dst

    retval, dst_otsu = cv2.threshold(src, 0, maxval, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results['THRESH_OTSU'] = dst_otsu

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(src, cmap='gray')
    plt.title('Исходное изображение')
    plt.axis('off')

    for i, (key, result) in enumerate(results.items(), start=2):
        plt.subplot(2, 2, i)
        plt.imshow(result, cmap='gray')
        plt.title(key)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def second():
    image = cv2.imread(adapt_threshold_image, cv2.IMREAD_GRAYSCALE)

    max_value = 255
    adaptive_methods = [cv2.ADAPTIVE_THRESH_MEAN_C, cv2.ADAPTIVE_THRESH_GAUSSIAN_C]
    block_sizes = [5, 10, 15, 30, 55]
    c_value = 10

    plt.figure(figsize=(12, 8))

    _, global_thresh = cv2.threshold(image, 0, max_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plt.subplot(2, len(block_sizes) + 1, 1)
    plt.imshow(global_thresh, cmap='gray')
    plt.title('Global Otsu Thresholding')
    plt.axis('off')

    for index, block_size in enumerate(block_sizes):
        if block_size % 2 == 0:
            block_size += 1

        for adaptive_method in adaptive_methods:
            adaptive_thresh = cv2.adaptiveThreshold(image, max_value, adaptive_method, cv2.THRESH_BINARY, block_size,
                                                    c_value)
            plt.subplot(2, len(block_sizes) + 1, index + 2)
            plt.imshow(adaptive_thresh, cmap='gray')
            title = 'Mean' if adaptive_method == cv2.ADAPTIVE_THRESH_MEAN_C else 'Gaussian'
            plt.title(f'Adaptive {title}\nBlock Size: {block_size}')
            plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    first()
    second()
    # third()
