import pandas as pd
import numpy as np
from sklearn import linear_model, model_selection
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Read the data set:
data = pd.read_csv("e-shop clothing 2008.csv", sep=";")

# From the data set I will use three attributes.
data = data[["page 1 (main category)", "price", "price 2"]]

print(data.head())

# The data I want to predict:
predict = "price 2"

# x returns a new data frame without 'predict' (training data)
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

# test_size = splitting 10% of the data in x_test & y_test
# x_train will be a section of the x array
# y_train will be a section of the y array
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.1)

# Training module
linear = linear_model.LinearRegression()
# finding a best fit line
linear.fit(x_train, y_train)
# Returns the value
acc = linear.score(x_test, y_test)
print(acc)

predictions = linear.predict(x_test) 

for x, val in enumerate(predictions):
  print  ('Prediction: ', round(predictions[x]), x_test[x], 'Official Result: ', y_test[x])
