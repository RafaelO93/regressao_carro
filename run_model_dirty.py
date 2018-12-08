# -*- coding: utf-8 -*-
import warnings

from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

dataset = pd.read_csv("true_car_listings.csv", header=0)

y_dataset = dataset[['Price']]
dataset = dataset[["Year", "Mileage", "State", "Make"]]
dataset = pd.get_dummies(dataset, columns=["State", "Make"])

X_train, X_test, y_train, y_test = train_test_split(dataset, y_dataset.values.ravel(), test_size=0.20, random_state=42)
warnings.filterwarnings("ignore", category=DeprecationWarning)

dt = DecisionTreeRegressor()
# dt = reg = LinearRegression()
# dt = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)


print("Treinamento")
print(dataset.shape)
dt_fit = dt.fit(X_train, y_train)

dt_scores = cross_val_score(dt_fit, X_train, y_train, cv=10)
dt_predict = dt.predict(X_test)
print("Media cross validation score: {}".format(np.mean(dt_scores)))
print("RMSE Score: ", sqrt(mean_squared_error(y_test, dt_predict)))

plt.scatter(dt_predict, y_test)
z = np.polyfit(dt_predict, y_test, 1)
p = np.poly1d(z)
plt.title('DecisionTree Resultados - Sem Normalizar')
plt.plot(y_test, p(y_test), "r--")
plt.xlabel('Valores Encontrados')
plt.ylabel('Valores Esperados')
plt.grid(True)
plt.show()
