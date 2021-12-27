import cv2
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass


@dataclass
class Point:
    x: int
    y: int
    useful: bool


points: list[Point] = list()


def main():
    imgOriginal = readImage()
    img = cleanImage(imgOriginal)
    imgDetected = detectLeaf(img)
    img = erodeImage(imgDetected)
    imgD = detectEdge(img)

    # minMax = findMinMax(imgD)
    #
    # for i in range(0, len(points)):
    #     if i < len(points) - 1:
    #         if (points[i].useful and points[i + 1].useful and points[i].y == points[i + 1].y) and (
    #                 points[i + 1].x - points[i].x < 2):
    #             points[i].useful = False
    #
    #     # if minMax[0] < points[i].x < minMax[1] and minMax[2] < points[i].y < minMax[3]:
    #     #     points[i].useful = False
    #
    # for p in points:
    #     if p.useful:
    #         img = cv2.circle(imgOriginal, (p.x, p.y), 0, (255, 0, 0), -1)
    #         # print("(", p.x, p.y, ")")
    #
    # # leafType = checkCuoriformi(img)
    #
    # # print("E' CUORIFORME? ", leafType)
    #
    # if checkLanceolata(img, minMax):
    #     print('FOGLIA LANCEOLATA')
    # else:
    #     if checkCuoriformi(img):
    #         print('FOGLIA CUORIFORME')

    plotImage(imgD)


def readImage():
    location = "./images/ceropegia.jpg"
    leaf = cv2.imread(location)
    leaf = cv2.cvtColor(leaf, cv2.COLOR_BGR2RGB)
    return leaf


def cleanImage(image):
    img = cv2.GaussianBlur(image, (31, 31), 0)
    return img


def detectEdge(image):
    # img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(image, 150, 300)
    indices = np.where(edges != [0])

    for i in range(0, len(indices[0])):
        points.append(Point(indices[1][i], indices[0][i], useful=True))

    # print(img.shape)
    # print(tuple(coordinates))
    return edges


def detectLeaf(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # find the brown color
    mask_brown = cv2.inRange(hsv, (8, 60, 20), (30, 255, 200))
    # find the yellow and green color in the leaf
    mask_yellow_green = cv2.inRange(hsv, (10, 39, 64), (86, 255, 255))
    # find any of the three colors(green or brown or yellow) in the image
    mask = cv2.bitwise_or(mask_yellow_green, mask_brown)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img, img, mask=mask)
    # res = cv2.cvtColor(res, cv2.COLOR_HSV2RGB)
    # imgGray = cv2.cvtColor(imgGray, cv2.COLOR_RGB2GRAY)
    # (thresh, im_bw) = cv2.threshold(imgGray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return res


def erodeImage(image):
    # return image
    kernel = np.ones(5)
    dilate = cv2.dilate(image, kernel, 1)
    return cv2.erode(dilate, kernel, iterations=1)
    #return cv2.erode(image, kernel, 1)


def plotImage(image):
    plt.imshow(image)
    plt.title("Quercia RGB")
    plt.show()


def checkCuoriformi(image):
    result = False
    maxY = points[len(points) - 1].y
    minY = maxY - (maxY / 4)  # CONSIDERIAMO SOLO L'ULTIMO QUARTO DI FOGLIA
    print(minY, maxY)

    counter = 0
    innerCounter = 0
    oldValue = 0
    for p in points:
        if p.y > minY:
            if oldValue != p.y:
                if innerCounter >= 4:
                    counter += 1
                oldValue = p.y
                innerCounter = 0
            else:
                innerCounter += 1

    print("COUNTER: ", counter)

    return counter > 30


def findMinMax(image):
    minX = image.shape[0]
    maxX = 0
    minY = image.shape[1]
    maxY = 0

    for p in points:
        if p.y > maxY:
            maxY = p.y

        if p.y < minY:
            minY = p.y

        if p.x > maxX:
            maxX = p.x

        if p.x < minX:
            minX = p.x

    return [minX, maxX, minY, maxY]


def checkLanceolata(image, minMax):
    width = minMax[1] - minMax[0]
    height = minMax[3] - minMax[2]

    print("height", height, "width", width)
    aspectRatio = width / height
    print("ASPECT RATIO", aspectRatio)

    return 0.32 <= aspectRatio <= 0.48


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)
