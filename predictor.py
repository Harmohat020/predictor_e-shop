import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow
import keras
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model


data = pd.read_csv("e-shop clothing 2008.csv", sep=";")

data = data[["page 1 (main category)", "price", "price 2"]]

print(data.head())

# # G3 = Eind Cijfer
predict = "price 2"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
# print(acc)

print('Co: ', linear.coef_)
print('Intercept: ', linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
  print('Prediction: ' , predictions[x], x_test[x], 'Official Result: ', y_test[x])