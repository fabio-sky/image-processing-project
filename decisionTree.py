import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


class LeafDetails:
    height: int
    width: int
    lobulata: bool
    cuoriforme: bool
    lanceolata: bool

    def __init__(self):
        self.height = 0
        self.width = 0
        self.lobulata = False
        self.cuoriforme = False
        self.lanceolata = False

    def clearAll(self):
        self.height = 0
        self.width = 0
        self.lobulata = False
        self.lanceolata = False
        self.cuoriforme = False


class DecisionTree:
    colNames = ["name", "height", "width", "lobulata", "cuoriforme", "lanceolata"]
    featuresCol = ["height", "width", "lobulata", "cuoriforme", "lanceolata"]
    classNames = ["Oleandro", "Olivo", "Quercia", "Magnolia", "Heuchera", "Ciclamino"]
    decisionTree: DecisionTreeClassifier

    def __init__(self):
        self.initializeTree()

    def initializeTree(self):
        dataset = pd.read_csv("./leaf_dataset.csv", header=0, names=self.colNames)
        dataset.head()

        X = dataset[self.featuresCol]
        y = dataset.name

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

        self.decisionTree = DecisionTreeClassifier(criterion="entropy")
        self.decisionTree = self.decisionTree.fit(X_train.values, y_train)
        y_pred = self.decisionTree.predict(X_test.values)

        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    def predictLeaf(self, data: LeafDetails):
        predictData = [[data.height, data.width, data.lobulata, data.cuoriforme, data.lanceolata]]
        prediction = self.decisionTree.predict(predictData)
        return prediction[0]
