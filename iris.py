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

# Load the Iris dataset from sklearn
iris = datasets.load_iris()

# 1. Number of times pregnant
# 2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
# 3. Diastolic blood pressure (mm Hg)
# 4. Triceps skin fold thickness (mm)
# 5. 2-Hour serum insulin (mu U/ml)
# 6. Body mass index (weight in kg/(height in m)^2)
# 7. Diabetes pedigree function
# 8. Age (years)
# 9. Class variable (0 or 1)

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']

# load dataset
pima = pd.read_csv("data/pima-indians.csv", header=None, names=col_names)

print(pima.head())

# Set up a pipeline with a feature selection preprocessor that
# selects the top 2 features to use.
# The pipeline then uses a RandomForestClassifier to train the model.

pipeline = Pipeline([
      ('feature_selection', SelectKBest(chi2, k=2)),
      ('classification', DecisionTreeClassifier())
    ])

#pipeline = Pipeline([
#      ('feature_selection', SelectKBest(chi2, k=2)),
#      ('classification', RandomForestClassifier())
#    ])

pipeline.fit(iris.data, iris.target)

# Export the classifier to a file
joblib.dump(pipeline, 'model.joblib')
