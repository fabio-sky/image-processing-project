import pandas as pd
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

colNames = ["name", "height", "width", "lobulata", "cuoriforme", "lanceolata"]


def main():
    print('MAIN')
    leaf = pd.read_csv("./leaf_dataset.csv", header=1, names=colNames)
    leaf.head()

    featuresCol = ["height", "width", "lobulata", "cuoriforme", "lanceolata"]
    X = leaf[featuresCol]
    y = leaf.name

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # clf = DecisionTreeClassifier(criterion="entropy")
    clf = DecisionTreeClassifier(criterion="entropy")
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True, feature_names=featuresCol, class_names=['Oleandro', 'Olivo'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('./leafDecisionTree.png')
    Image(graph.create_png())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)
