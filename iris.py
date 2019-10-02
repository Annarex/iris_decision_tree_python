import pandas as pd
from sklearn import datasets
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

from sklearn.externals import joblib

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus
import numpy as np
import array
import matplotlib.pyplot as plt

import os
import glob

# Load the Iris dataset from sklearn
#iris = datasets.load_iris()
#print(iris)

# 1. "sepal_length"
# 2. "sepal_width"
# 3. "petal_length"
# 4. "petal_width"
# 5. "variety"

col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'variety']

# load dataset
iris = pd.read_csv("data/input/iris.csv", header=None, names=col_names)

#print(pima.head())

#split dataset in features and target variable
feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X = iris[feature_cols] # Features
y = iris.variety # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test



criterion = ["entropy","gini"]
depth = 4
entropy_accuracy_array = []
gini_accuracy_array = []

path = '/home/haliax/Projects/iris_decision_tree_python'

files = glob.glob('{}/data/output/iris/*'.format(path))
for f in files:
    os.remove(f)

files = glob.glob('{}/models/iris/*'.format(path))
for f in files:
    os.remove(f)

for crit in criterion:
    for max_depth_var in range(1,depth + 1):

        # Create Decision Tree classifer object
        clf = DecisionTreeClassifier(criterion=crit, max_depth=max_depth_var)

        # Train Decision Tree Classifer
        clf = clf.fit(X_train,y_train)

        # Export the classifier to a file
        joblib.dump(clf, 'models/iris/iris-model-{}-depth{}.joblib'.format(crit,max_depth_var))

        #Predict the response for test dataset
        y_pred = clf.predict(X_test)

        # Model Accuracy, how often is the classifier correct?
        accuracy = metrics.accuracy_score(y_test, y_pred).round(3)
        print("Accuracy for iris model with criterion {} and max depth {}:".format(crit,max_depth_var), accuracy)

        entropy_accuracy_array.append(accuracy) if crit == "entropy" else gini_accuracy_array.append(accuracy)

        # Show the tree
        dot_data = StringIO()
        export_graphviz(clf, out_file=dot_data, filled=True, rounded=True,special_characters=True, feature_names = feature_cols,class_names=['Setosa','Versicolor','Virginica'])
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph.write_png('data/output/iris/iris-{}-depth{}.png'.format(crit,max_depth_var))
        Image(graph.create_png())
        #end depth for loop

depth_array = range(1,depth + 1)
print(depth_array)
print(entropy_accuracy_array)
print(gini_accuracy_array)

plt.plot(depth_array, entropy_accuracy_array, color='g')
plt.plot(depth_array, gini_accuracy_array, color='orange')
plt.xlabel('Tree Depth')
plt.ylabel('Accuracy')
plt.title('Entropy (green) vs Gini (orange) accuracy vs tree depth')
plt.show()

#----------------------------------------------------------------------------------------------------------------------------
# Set up a pipeline with a feature selection preprocessor that
# selects the top 2 features to use.
# The pipeline then uses a RandomForestClassifier to train the model.

#pipeline = Pipeline([
#      ('feature_selection', SelectKBest(chi2, k=2)),
#      ('classification', DecisionTreeClassifier())
#    ])

#pipeline.fit(iris.data, iris.target)
