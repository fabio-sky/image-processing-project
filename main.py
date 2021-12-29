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
pointsByX: list[Point] = list()

leafName = 'oak_red.jpg'


def main():
    imgOriginal = readImage()
    # imgRotated = rotateImage(imgOriginal)
    img = cleanImage(imgOriginal)
    imgDetected = detectLeaf(img)
    img = erodeImage(imgDetected)
    imgD = detectEdge(img)

    print('IMG SHAPE', imgD.shape)

    minMax = findMinMax(imgD)

    # Non consideriamo i punti attaccati
    for i in range(0, len(points)):
        if i < len(points) - 1:
            if (points[i].useful and points[i + 1].useful and points[i].y == points[i + 1].y) and (
                    points[i + 1].x - points[i].x < 2):
                points[i].useful = False

    for i in range(0, len(pointsByX)):
        if i < len(pointsByX) - 1:
            if (pointsByX[i].useful and pointsByX[i + 1].useful and pointsByX[i].x == pointsByX[i + 1].x) and (
                    pointsByX[i + 1].y - pointsByX[i].y < 2):
                pointsByX[i].useful = False

    for p in points:
        # print("(", p.x, p.y, ")")
        if p.useful:
            img = cv2.circle(imgOriginal, (p.x, p.y), 2, (255, 0, 0), -1)

    for p in pointsByX:
        print("(", p.x, p.y, p.useful, ")")
        if p.useful:
            img = cv2.circle(img, (p.x, p.y), 2, (0, 0, 255), -1)


    title = 'FOGLIA'

    if checkLanceolata(minMax):
        title += ' LANCEOLATA '
    if checkLobulate(imgD.shape[1]):
        title += ' LOBULATA '
    if checkCuoriformi(minMax):
        title += ' CUORIFORME '

    plotImage(imgOriginal, title)


def readImage():
    location = "./images/" + leafName
    leaf = cv2.imread(location)
    leaf = cv2.cvtColor(leaf, cv2.COLOR_BGR2RGB)
    return leaf


def cleanImage(image):
    img = cv2.GaussianBlur(image, (21, 21), 0)
    return img


def sortPointsByX(p: Point):
    return p.x


def detectEdge(image):
    # img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(image, 150, 300)
    indices = np.where(edges != [0])

    for i in range(0, len(indices[0])):
        points.append(Point(indices[1][i], indices[0][i], useful=True))
        pointsByX.append(Point(indices[1][i], indices[0][i], useful=True))

    pointsByX.sort(key=sortPointsByX)

    # print(img.shape)
    # print(tuple(coordinates))
    return edges


def toHsvOpencvRange(h, s, v):
    hOpen = (h * 179) / 360
    sOpen = (s * 255) / 100
    vOpen = (v * 255) / 100

    return hOpen, sOpen, vOpen


def detectLeaf(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # find the brown color
    mask_red_brown = cv2.inRange(hsv, toHsvOpencvRange(5, 23, 7), toHsvOpencvRange(60, 80, 78))
    # find the yellow and green color in the leaf
    mask_yellow_green = cv2.inRange(hsv, toHsvOpencvRange(20, 6, 25), toHsvOpencvRange(172, 100, 100))
    # find any of the three colors(green or brown or yellow) in the image
    mask = cv2.bitwise_or(mask_yellow_green, mask_red_brown)
    # mask = cv2.bitwise_or(mask, mask_dark_green)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img, img, mask=mask)
    # res = cv2.cvtColor(res, cv2.COLOR_HSV2RGB)
    # imgGray = cv2.cvtColor(imgGray, cv2.COLOR_RGB2GRAY)
    # (thresh, im_bw) = cv2.threshold(imgGray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return res


def erodeImage(image):
    # return image
    kernel = np.ones(10)
    kernelSmall = np.ones(5)
    # kernel = generateCircle(10)
    # print(kernel)
    # return image
    erode = cv2.erode(image, kernelSmall, iterations=1)
    dilate = cv2.dilate(erode, kernel, 1)
    return cv2.erode(dilate, kernel, iterations=1)


def plotImage(image, title):
    plt.imshow(image)
    plt.title(title)
    plt.show()


def checkCuoriformi(minMax):
    result = False
    # maxY = pointsByX[len(pointsByX) - 1].y
    # minY = maxY - (maxY / 4)  # CONSIDERIAMO SOLO L'ULTIMO QUARTO DI FOGLIA
    minY = minMax[3] - (minMax[3] / 4)  # CONSIDERIAMO SOLO L'ULTIMO QUARTO DI FOGLIA
    print(minY, minMax[3])

    counter = 0
    innerCounter = 0
    oldValue = 0
    for p in points:
        if p.y > minY and p.useful:
            if oldValue != p.y:
                if innerCounter >= 4:
                    counter += 1
                oldValue = p.y
                innerCounter = 1
            else:
                innerCounter += 1

    print("COUNTER CUORIFORME: ", counter)

    return counter > 30


def rotateImage(image):
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)


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


def checkLanceolata(minMax):
    width = minMax[1] - minMax[0]
    height = minMax[3] - minMax[2]

    print("height", height, "width", width)
    aspectRatio = width / height
    print("ASPECT RATIO", aspectRatio)

    return 0.1 <= aspectRatio <= 0.48


def checkLobulate(imageWidth):
    # image[yMin:yMax, xMin, xMax]

    counterSX = 0
    innerCounter = 0
    oldValue = 0
    for p in pointsByX:
        if p.x < int(imageWidth / 2) and p.useful:  # consideriamo la metà SX
            if oldValue != p.x:
                if innerCounter >= 6:
                    counterSX += 1
                oldValue = p.x
                innerCounter = 1
            else:
                innerCounter += 1

    counterDX = 0
    innerCounter = 0
    oldValue = 0
    for p in pointsByX:
        if p.x > int(imageWidth / 2) and p.useful:  # consideriamo la metà DX
            if oldValue != p.x:
                if innerCounter >= 6:
                    counterDX += 1
                oldValue = p.x
                innerCounter = 1
            else:
                innerCounter += 1

    print("COUNTER: ", counterSX, counterDX)

    return counterDX > 10 or counterSX > 10


def unique_count_app(image):
    colors, count = np.unique(image.reshape(-1, image.shape[-1]), axis=0, return_counts=True)
    maxIndex = count.argmax()
    ret = colors[maxIndex]
    if ret[0] == 0 and ret[1] == 0 and ret[2] == 0:
        print('SEEEEEE')
        np.delete(colors, maxIndex)
        np.delete(count, maxIndex)
        maxIndex = count.argmax()
        ret = colors[maxIndex]
    print('COLOR', count, ret, maxIndex)
    # return ret


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)
