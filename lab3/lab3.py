import cv2
import matplotlib.pyplot as plt

def first():
# show original image
    img = cv2.imread('image3.jpg')
    cv2.imshow('Original Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # show grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('1_1_gray.jpg', gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # calculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    plt.hist(hist, 256, [0, 256])
    plt.title('Гистограмма серого')
    plt.show()


    # show normalized image
    gray_norm = cv2.normalize(gray, None, 63, 255, cv2.NORM_MINMAX)
    cv2.imshow('1_normalized.jpg', gray_norm)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # show normalized histogram
    hist_norm = cv2.calcHist([gray_norm], [0], None, [256], [0, 256])
    plt.hist(hist_norm, 256, [0, 256])
    plt.title('гистограмма серого нормализированного изображения')
    plt.show()

def second():
    img = cv2.imread('color4.jpg')
    cv2.imshow('Original Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('grey_2_1.jpg', gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    plt.hist(hist, 256, [0, 256])
    plt.title('Гистограмма серого')
    plt.show()

    gray_eq = cv2.equalizeHist(gray)
    cv2.imshow('equalized_2_2.jpg', gray_eq)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    hist_eq = cv2.calcHist([gray_eq], [0], None, [256], [0, 256])
    plt.hist(hist_eq, 256, [0, 256])
    plt.title('Гистограмма серого после эквализации')
    plt.show()

def third():
    img = cv2.imread('image_gray_63-192.jpg', cv2.IMREAD_GRAYSCALE)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('grey_3_1.jpg', gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    plt.hist(hist, 256, [0, 256])
    plt.title('Гистограмма серого')
    plt.show()

    gray_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imshow('grey_norm_3_2.jpg', gray_norm)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    hist_norm = cv2.calcHist([gray_norm], [0], None, [256], [0, 256])
    plt.hist(hist_norm, 256, [0, 256])
    plt.title('гистограмма серого пеосле нормализации изображения')
    plt.show()

    gray_eq = cv2.equalizeHist(gray)
    cv2.imshow('equalized_3_3.jpg', gray_eq)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    hist_eq = cv2.calcHist([gray_eq], [0], None, [256], [0, 256])
    plt.hist(hist_eq, 256, [0, 256])
    plt.title('гистограмма серого после эквализации изображения')
    plt.show()



if __name__ == '__main__':
    first()
    second()
    third()