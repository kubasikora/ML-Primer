import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
from sklearn import linear_model
from sklearn.metrics import r2_score

# load data
df = pd.read_csv("./data/fuel_consumption.csv")
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

# show data histograms
# viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
# viz.hist()
# plt.show()

# plot relation between two columns
# plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
# plt.xlabel("FUELCONSUMPTION_COMB")
# plt.ylabel("Emission")
# plt.show()

# create train and test dataset
mask = np.random.rand(len(df)) < 0.8
train_set = cdf[mask]
test_set = cdf[~mask]

# create model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train_set[['ENGINESIZE']])
train_y = np.asanyarray(train_set[['CO2EMISSIONS']])
regr.fit(train_x, train_y)

# print coefficients
print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)

# eval model metrics
test_x = np.asanyarray(test_set[['ENGINESIZE']])
test_y = np.asanyarray(test_set[['CO2EMISSIONS']])
model_test_result = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(model_test_result - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((model_test_result - test_y) ** 2))
print("R2-score: %.2f" % r2_score(model_test_result, test_y))

# plot model over the data
plt.scatter(train_set.ENGINESIZE, train_set.CO2EMISSIONS, color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()