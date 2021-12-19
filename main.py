import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():
    print('MAIN')
    img = readImage()
    img = cleanImage(img)
    img = detectLeaf(img)
    img = detectEdge(img)
    plotImage(img)


def readImage():
    print('READ IMAGE')
    location = "./images/quercia.jpg"
    leaf = cv2.imread(location)
    leaf = cv2.cvtColor(leaf, cv2.COLOR_BGR2RGB)
    return leaf


def cleanImage(image):
    print('CLEAN IMAGE')
    img = cv2.GaussianBlur(image, (5, 5), 0)
    return img


def detectEdge(image):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(img, 150, 300)
    indices = np.where(edges != [0])
    coordinates = zip(indices[0], indices[1])
    print(indices[0])
    return edges

def detectLeaf(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # find the brown color
    mask_brown = cv2.inRange(hsv, (8, 60, 20), (30, 255, 200))
    # find the yellow and green color in the leaf
    mask_yellow_green = cv2.inRange(hsv, (10, 39, 64), (86, 255, 255))
    # find any of the three colors(green or brown or yellow) in the image
    mask = cv2.bitwise_or(mask_yellow_green, mask_brown)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img, img, mask=mask)
    return res


def plotImage(image):
    plt.subplot(2, 2, 1).axis('off')
    plt.imshow(image)
    plt.title("Quercia RGB")
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)
