import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree  # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation

colNames = ["name", "height", "width", "lobulata", "cuoriforme", "lanceolata"]


def main():
    print('MAIN')
    leaf = pd.read_csv("./leaf_dataset.csv", header=1, names=colNames)
    leaf.head()

    featuresCol = ["height", "width", "lobulata", "cuoriforme", "lanceolata"]
    X = leaf[featuresCol]
    y = leaf.name

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

    # clf = DecisionTreeClassifier(criterion="entropy")
    print( X_test)
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    fig = plt.figure(figsize=(25, 20))
    _ = plot_tree(clf,
                  feature_names=featuresCol,
                  class_names=["Oleandro", "Olivo", "Quercia", "Magnolia", "Heuchera", "Ciclamino"],
                  filled=True)
    fig.savefig("decistion_tree_gini.png")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)
