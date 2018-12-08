# -*- coding: utf-8 -*-
from math import sqrt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import NearestNeighbors

dataset = pd.read_csv("true_car_listings_clean.csv", header=0).head(100)
dataset = dataset.dropna()
dataset = dataset.drop(dataset.columns[0], axis=1)

y_dataset = dataset[['Price']]
dataset = dataset.loc[:, dataset.columns != 'Price']

dataset = pd.get_dummies(dataset, columns=["Region"])

X_train, X_test, y_train, y_test = train_test_split(dataset, y_dataset.values.ravel(), test_size=0.20, random_state=42)

# models = [DecisionTreeRegressor(), LinearRegression(),
#           RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100), NearestNeighbors(n_neighbors=10)]

models = [SGDRegressor()]
for model in models:
    dt = model
    print("Treinamento")
    print(dataset.shape)

    dt_scores = cross_val_score(dt.fit(X_train, y_train), X_train, y_train, cv=10)
    dt_predict = dt.predict(X_test)
    print("Media cross validation score: {}".format(np.mean(dt_scores)))
    print("RMSE Score: ", sqrt(mean_squared_error(y_test, dt_predict)))
    print("--------------------------------------------------")
    plt.scatter(dt_predict, y_test)
    print (dt_predict)
    print(y_test)
    print(np.subtract(dt_predict, y_test))
    plt.title('SGDRegressor Resultados - Sem Normalizar')
    z = np.polyfit(dt_predict, y_test, 1)
    p = np.poly1d(z)
    plt.plot(y_test, p(y_test), "r--")
    plt.grid(True)
    plt.show()
