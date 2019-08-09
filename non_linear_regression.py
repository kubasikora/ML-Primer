import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

df = pd.read_csv("./data/china_gdp.csv")

# split data
x_data, y_data = (df["Year"].values, df["Value"].values)


# normalize data
xdata = x_data/max(x_data)
ydata = y_data/max(y_data)

# model function
def sigmoid(x, Beta_1, Beta_2):
    return 1 / (1 + np.exp(-Beta_1*(x - Beta_2)))

# split data into train and test set
mask = np.random.rand(len(df)) < 0.8
train_set_x = xdata[mask]
train_set_y = ydata[mask]
test_set_x = xdata[~mask]
test_set_y = ydata[~mask]

# fit model
popt, pcov = curve_fit(sigmoid, train_set_x, train_set_y)

# print optimized parameters
print("Beta_1 = %f, Beta_2 = %f" % (popt[0], popt[1]))

# eval model metrics
y_hat = sigmoid(test_set_x, *popt)

print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat - test_set_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat - test_set_y) ** 2))
print("R2-score: %.2f" % r2_score(y_hat , test_set_y) )

# plot model over the data
x = np.linspace(1960, 2015, 55)
x = x/max(x)
plt.figure(figsize=(8,5))
y = sigmoid(x, *popt)
plt.plot(test_set_x, test_set_y, 'ro', label='data')
plt.plot(x,y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()