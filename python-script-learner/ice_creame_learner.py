import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT_PATH = os.path.abspath(os.path.join(".."))
DATA_FILE_PATH = os.path.join(ROOT_PATH, "data-file", "ice_cream_rater_data.csv")
OUTPUT_MODEL_PATH = os.path.join(ROOT_PATH, "output-model", "ice_cream_rater_model.joblib")

try:
    data = pd.read_csv(DATA_FILE_PATH)
    X = data.drop('rating', axis=1)
    y = data.loc[:, 'rating']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8, random_state=100)

    y_train = np.round(y_train)
    y_test = np.round(y_test)


    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", KNeighborsClassifier())
    ])
    model1 = GridSearchCV(estimator=pipe,  # an estimator has to be something that has a dot fit and predict
                        param_grid={'model__n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
                        # param_grid can be set over any paramerter from the estimator
                        # we can run the estimator get_param function and see all params
                        cv=3  # this set how much to divide the training data (train sections and test section)
                        )

    model1.fit(X_train, y_train)
    print(f"Knn: {model1.score(X_test, y_test)}")
    joblib.dump(model1, filename=OUTPUT_MODEL_PATH)
except Exception as exp:
    sys.stderr.write(f"Exception occured while training the model: {str(exp)}")
    sys.exit(1)
sys.exit(0)
