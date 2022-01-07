import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree  # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation


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

    # def setHeight(self, value):
    #     self.height = value
    #
    # def setWidth(self, value):
    #     self.width = value
    #
    # def setLobulata(self, value):
    #     self.lobulata = value
    #
    # def setCuoriforme(self, value):
    #     self.cuoriforme = value
    #
    # def setLanceolata(self, value):
    #     self.lanceolata = value

    def clearAll(self):
        self.height = 0
        self.width  = 0
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
        dataset = pd.read_csv("./leaf_dataset_2.csv", header=0, names=self.colNames)
        dataset.head()

        X = dataset[self.featuresCol]
        y = dataset.name

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

        self.decisionTree = DecisionTreeClassifier(criterion="entropy")
        self.decisionTree = self.decisionTree.fit(X_train, y_train)
        y_pred = self.decisionTree.predict(X_test)

        # self.printDecisionTree()
        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    def predictLeaf(self, data: LeafDetails):
        print("LEAF DETAILS", data.lanceolata, data.lobulata, data.cuoriforme)
        predictData = [[data.height, data.width, data.lobulata, data.cuoriforme, data.lanceolata]]
        prediction = self.decisionTree.predict(predictData)
        return prediction[0]

    def printDecisionTree(self):
        fig = plt.figure(figsize=(25, 20))
        _ = plot_tree(self.decisionTree,
                      feature_names=self.featuresCol,
                      class_names=self.classNames,
                      filled=True)
        fig.savefig("decision_tree_2.png")
