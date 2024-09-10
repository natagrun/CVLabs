import cv2 as cv
import numpy as np

if __name__ == '__main__':
    img = np.zeros((400, 400, 3), dtype="uint8")
    cv.imshow('df', img)
    cv.waitKey(1000)

    img = np.zeros((400, 400, 3), dtype="uint8")
    cv.line(img, (15, 20), (70, 50), color=(0, 255, 0), thickness=2)
    cv.imshow('df', img)
    cv.waitKey(1000)

    img = np.zeros((400, 400, 3), dtype="uint8")
    cv.circle(img, (200, 200), 32, color=(255, 0, 0))
    cv.imshow('df', img)
    cv.waitKey(1000)

    img = np.zeros((400, 400, 3), dtype="uint8")
    cv.ellipse(img, (200, 200), (100, 160), angle=45, startAngle=0, endAngle=360, color=(0, 0, 255))
    cv.imshow('df', img)
    cv.waitKey(1000)

    img = np.zeros((400, 400, 3), dtype="uint8")
    cv.rectangle(img, (15, 20), (70, 50), color=(0, 255, 255))
    cv.imshow('df', img)
    cv.waitKey(1000)

    polygonePts = np.array([[110, 350], [130, 350],
                            [150, 200], [110, 200], [110, 100],
                            [140, 100], [140, 150], [170, 150],
                            [170, 100], [200, 100], [200, 150],
                            [230, 150], [230, 100], [260, 100],
                            [260, 200], [220, 200], [240, 350],
                            [260, 350], [260, 390], [110, 390]])

    img = np.zeros((400, 400, 3), dtype="uint8")
    cv.fillPoly(img, [polygonePts], color=(255, 255, 255))
    cv.imshow('df', img)
    cv.waitKey(1000)

    img = np.zeros((400, 400, 3), dtype="uint8")
    cv.polylines(img, [polygonePts], isClosed=True, color=(255, 255, 255))
    cv.imshow('df', img)
    cv.waitKey(1000)

    img = np.zeros((400, 400, 3), dtype="uint8")
    cv.polylines(img, [polygonePts], isClosed=False, color=(255, 255, 255))
    cv.imshow('df', img)
    cv.waitKey(1000)

    img = np.zeros((400, 400, 3), dtype="uint8")
    cv.putText(img, "Hello, word!", (20, 375), cv.FONT_ITALIC, 1, (255, 0, 255), 1)
    cv.putText(img, "Hello, word!", (80, 50), cv.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (255, 255, 0), 1)
    cv.imshow('df', img)
    cv.waitKey(1000)

    cv.destroyAllWindows()
