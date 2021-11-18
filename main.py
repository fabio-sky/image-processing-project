import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():
    print('MAIN')
    readImage()


def readImage():
    print('READ IMAGE')
    location = "./images/quercia.jpg"
    leaf = cv2.imread(location)
    leaf = cv2.cvtColor(leaf, cv2.COLOR_BGR2RGB)
    cleanImage()
    plotImage(leaf)


def cleanImage():
    print('CLEAN IMAGE')


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
