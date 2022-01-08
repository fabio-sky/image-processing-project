import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import tkinter.messagebox as messagebox
from dataclasses import dataclass
from decisionTree import DecisionTree, LeafDetails


@dataclass
class Point:
    x: int
    y: int
    useful: bool


# ---------------------------------------
# GLOBAL VARIABLES

selectedLeaf: tk.IntVar
leafImage: tk.Label

pointsY: list[Point] = list()
pointsX: list[Point] = list()

path: str = ""

leafDecision = DecisionTree()
leafData = LeafDetails()


# ------------------------------------------------------

def main():
    imgOriginal = readImage(path)
    imgBlur = blurImage(imgOriginal)
    imgDetected = detectLeaf(imgBlur)
    imgFilled = fillHoleImage(imgDetected)
    imgEdge = detectEdge(imgFilled)

    filterUsefulPoints()

    minMax = findMinMax(imgEdge)
    img = printBorder(imgFilled)

    classifyLeaf(minMax)

    leafPrediction = leafDecision.predictLeaf(leafData)

    plotImage(img, leafPrediction)


def readImage(pth):
    img = cv2.imread(pth)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def blurImage(image):
    img = cv2.GaussianBlur(image, (21, 21), 0)
    return img


def detectEdge(image):
    def sortPointsByX(p: Point):
        return p.x

    edges = cv2.Canny(image, 150, 300)
    indices = np.where(edges != [0])

    for i in range(0, len(indices[0])):
        pointsY.append(Point(indices[1][i], indices[0][i], useful=True))
        pointsX.append(Point(indices[1][i], indices[0][i], useful=True))

    pointsX.sort(key=sortPointsByX)
    return edges


def detectLeaf(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    maskRedBrown = cv2.inRange(hsv, toHsvOpencvRange(5, 20, 7), toHsvOpencvRange(60, 80, 78))
    maskYellowGreen = cv2.inRange(hsv, toHsvOpencvRange(20, 4, 15), toHsvOpencvRange(172, 100, 85))
    maskGrey = cv2.inRange(hsv, toHsvOpencvRange(0, 0, 15), toHsvOpencvRange(360, 15, 30))

    mask = cv2.bitwise_or(maskYellowGreen, maskRedBrown)
    mask = cv2.bitwise_or(mask, maskGrey)

    res = cv2.bitwise_and(img, img, mask=mask)
    return res


def fillHoleImage(image):
    kernel = np.ones(20)
    dilate = cv2.dilate(image, kernel, 1)
    return cv2.erode(dilate, kernel, iterations=1)


# --------------------------------------------------------------------------------
# UTILS FUNCTIONS

# Trova i punti agli estremi della foglia
def findMinMax(image):
    minX = image.shape[0]
    maxX = 0
    minY = image.shape[1]
    maxY = 0

    for p in pointsY:
        if p.y > maxY:
            maxY = p.y

        if p.y < minY:
            minY = p.y

        if p.x > maxX:
            maxX = p.x

        if p.x < minX:
            minX = p.x

    return [minX, maxX, minY, maxY]


def plotImage(image, leafPrediction):
    title = 'Foglia'

    if leafData.lanceolata:
        title += ' LANCEOLATA '
    if leafData.lobulata:
        title += ' LOBULATA '
    if leafData.cuoriforme:
        title += ' CUORIFORME '
    title += " | " + leafPrediction

    plt.figure(num="Risultato")
    plt.imshow(image)
    plt.title(title)

    plt.show()


def toHsvOpencvRange(h, s, v):
    hOpen = (h * 179) / 360
    sOpen = (s * 255) / 100
    vOpen = (v * 255) / 100

    return hOpen, sOpen, vOpen


def printBorder(image):
    img = image
    for p in pointsX:
        img = cv2.circle(img, (p.x, p.y), 3, (255, 0, 255), -1)

    return img


def filterUsefulPoints():
    # Non consideriamo i punti attaccati
    for i in range(0, len(pointsY)):
        if i < len(pointsY) - 1:
            if (pointsY[i].useful and pointsY[i + 1].useful and pointsY[i].y == pointsY[i + 1].y) and (
                    pointsY[i + 1].x - pointsY[i].x < 2):
                pointsY[i].useful = False

    for i in range(0, len(pointsX)):
        if i < len(pointsX) - 1:
            if (pointsX[i].useful and pointsX[i + 1].useful and pointsX[i].x == pointsX[i + 1].x) and (
                    pointsX[i + 1].y - pointsX[i].y < 2):
                pointsX[i].useful = False


# ------------------------------------------------------------------
# CLASSIFICATION FUNCTIONS

def checkLobulate(leafWidth):
    counterSX = 0
    innerCounter = 0
    oldValue = 0
    for p in pointsX:
        if p.x < int(leafWidth / 2) and p.useful:  # consideriamo la metà SX
            if oldValue != p.x:
                if innerCounter >= 4:
                    counterSX += 1
                oldValue = p.x
                innerCounter = 1
            else:
                innerCounter += 1

    counterDX = 0
    innerCounter = 0
    oldValue = 0
    for p in pointsX:
        if p.x > int(leafWidth / 2) and p.useful:  # consideriamo la metà DX
            if oldValue != p.x:
                if innerCounter >= 4:
                    counterDX += 1
                oldValue = p.x
                innerCounter = 1
            else:
                innerCounter += 1

    sogliaMinima = int(leafWidth * 10 / 250)
    return counterDX >= sogliaMinima and counterSX >= sogliaMinima


def checkLanceolata(minMax):
    width = minMax[1] - minMax[0]
    height = minMax[3] - minMax[2]

    aspectRatio = width / height

    return 0.1 <= aspectRatio <= 0.48


def checkCuoriformi(minMax):
    minY = minMax[3] - (minMax[3] / 4)  # Consideriamo ultimo quarto di foglia

    counter = 0
    innerCounter = 0
    oldValue = 0
    for p in pointsY:
        if p.y > minY and p.useful:
            if oldValue != p.y:
                if innerCounter >= 4:
                    counter += 1
                oldValue = p.y
                innerCounter = 1
            else:
                innerCounter += 1

    leafWidth = minMax[1] - minMax[0]
    sogliaMinima = int(leafWidth * 50 / 450)

    return counter > sogliaMinima


def classifyLeaf(minMax):
    if checkLanceolata(minMax):
        leafData.lanceolata = True
    if checkLobulate(minMax[1] - minMax[0]):
        leafData.lobulata = True
    if checkCuoriformi(minMax):
        leafData.cuoriforme = True


# -------------------------------------------------------------------------------
# GUI FUNCTIONS


def selectLeaf():
    data = [[142, 21, "oleandro_2.jpg"], [70, 15, "olivo.jpg"], [176, 68, "magnolia_3.jpeg"],
            [55, 57, "heuchera_2.jpg"], [95, 51, "quercia_3.jpeg"], [101, 62, "quercia_4.jpg"],
            [40, 40, "ciclamino.jpeg"], ]

    sel = selectedLeaf.get()

    leafData.clearAll()

    leafData.height = data[sel - 1][0]
    leafData.width = data[sel - 1][1]
    global path
    path = "./images/" + data[sel - 1][2]


def startFlow():
    if selectedLeaf.get() > 0:
        pointsY.clear()
        pointsX.clear()
        main()
    else:
        messagebox.showwarning("Attenzione", "Selezionare una foglia da testare")


def showDecisionTree():
    decisionImg = readImage("./decision_tree.png")
    plt.figure(num="Decision Tree")
    plt.imshow(decisionImg)
    plt.title("Decision Tree")
    plt.axis('off')
    plt.show()


def initializeGUI():
    w = tk.Tk()
    w.geometry("250x310")
    w.title("Cielo Fabio - s292464")
    w.resizable(False, False)

    global selectedLeaf
    selectedLeaf = tk.IntVar()

    tk.Label(w, text="Scegli una foglia da testare", height=2).pack(anchor=tk.CENTER)

    tk.Radiobutton(w, text="Oleandro (142x21)", pady=4, value=1, variable=selectedLeaf, command=selectLeaf).pack(
        anchor=tk.W)

    tk.Radiobutton(w, text="Olivo (70x15)", pady=4, value=2, variable=selectedLeaf, command=selectLeaf).pack(
        anchor=tk.W)

    tk.Radiobutton(w, text="Magnolia (176x68)", pady=4, value=3, variable=selectedLeaf, command=selectLeaf).pack(
        anchor=tk.W)

    tk.Radiobutton(w, text="Heuchera (55x57)", pady=4, value=4, variable=selectedLeaf, command=selectLeaf).pack(
        anchor=tk.W)

    tk.Radiobutton(w, text="Quercia Marrone (31x51)", pady=4, value=5, variable=selectedLeaf, command=selectLeaf).pack(
        anchor=tk.W)

    tk.Radiobutton(w, text="Quercia Verde (101x62)", pady=4, value=6, variable=selectedLeaf, command=selectLeaf).pack(
        anchor=tk.W)

    tk.Radiobutton(w, text="Ciclamino (40x40)", pady=4, value=7, variable=selectedLeaf, command=selectLeaf).pack(
        anchor=tk.W)

    tk.Button(text="Analizza Foglia", command=startFlow).pack(anchor=tk.CENTER)

    tk.Button(text="Visualizza Decision Tree", command=showDecisionTree).pack(anchor=tk.CENTER)

    return w


# ------------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        window = initializeGUI()
        window.mainloop()
    except KeyboardInterrupt:
        exit(0)
