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

# Create Decision Tree classifer object

criterion = ["entropy","gini"]

for crit in criterion:
    for max_depth_var in range(1,10):
        clf = DecisionTreeClassifier(criterion=crit, max_depth=max_depth_var)

        # Train Decision Tree Classifer
        clf = clf.fit(X_train,y_train)

        # Export the classifier to a file
        joblib.dump(clf, 'models/iris/iris-model-{}-depth{}.joblib'.format(crit,max_depth_var))

        #Predict the response for test dataset
        y_pred = clf.predict(X_test)

        # Model Accuracy, how often is the classifier correct?
        print("Accuracy for iris model with criterion {} and max depth {}:".format(crit,max_depth_var),metrics.accuracy_score(y_test, y_pred))

        # Show the tree
        dot_data = StringIO()
        export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['Setosa','Versicolor','Virginica'])
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph.write_png('data/output/iris/iris-{}-depth{}.png'.format(crit,max_depth_var))
        Image(graph.create_png())





#----------------------------------------------------------------------------------------------------------------------------
# Set up a pipeline with a feature selection preprocessor that
# selects the top 2 features to use.
# The pipeline then uses a RandomForestClassifier to train the model.

#pipeline = Pipeline([
#      ('feature_selection', SelectKBest(chi2, k=2)),
#      ('classification', DecisionTreeClassifier())
#    ])

#pipeline.fit(iris.data, iris.target)
