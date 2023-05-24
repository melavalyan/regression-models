# Simple linear regression
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# input data
bill = [34.00, 108.00, 64.00, 88.00, 99.00, 51.00]
tip = [5.00, 17.00, 11.00, 8.00, 14.00, 5.00]

# plot the data points
plt.scatter(bill, tip)
plt.xlabel('Հաշիվը ($)')
plt.ylabel('Թեյավճարը ($)')

# calculate and plot the regression line
slope, intercept, r_value, p_value, std_err = stats.linregress(bill, tip)
x = np.linspace(min(bill), max(bill), 100)
y = slope * x + intercept
plt.plot(x, y, color='r')

# show the plot
plt.show()


import numpy as np

# Input data
bill = np.array([34.0, 108.0, 64.0, 88.0, 99.0, 51.0])
tip = np.array([5.0, 17.0, 11.0, 8.0, 14.0, 5.0])

# Calculate slope and intercept of the regression line
slope, intercept = np.polyfit(bill, tip, 1)

# Predict the tip for a bill amount of $70
bill_pred = 70
tip_pred = intercept + slope * bill_pred

print(f"Predicted tip for a bill amount of ${bill_pred}: ${tip_pred:.2f}")


# Multiple linear regression
import pandas as pd
from sklearn.linear_model import LinearRegression

# create a dataframe with the given data
data = {
    'road_length': [89, 66, 78, 111, 44, 77, 80, 66, 109, 76],
    'delivery_count': [4, 1, 3, 6, 1, 3, 3, 2, 5, 3],
    'time_spent': [7, 5.4, 6.6, 7.4, 4.8, 6.4, 7, 5.6, 7.3, 6.4]
}
df = pd.DataFrame(data)

# separate the features (x) and target variable (y)
x = df[['road_length', 'delivery_count']]
y = df['time_spent']

# fit a multiple linear regression model
model = LinearRegression().fit(x, y)

# print the coefficients
print(f"Coefficients: {model.intercept_} + {model.coef_[0]}*x1 + {model.coef_[1]}*x2")

# predict the travel time for a new shipment with the following characteristics
# road length: 95 km, delivery count: 2, petrol price: 3700 AMD
new_data = {'road_length': [70], 'delivery_count': [10]}
new_df = pd.DataFrame(new_data)
prediction = model.predict(new_df)
print(f"Predicted travel time for new shipment: {prediction[0]} hours")


# Bayesian linear regression

import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from theano import shared, tensor as tt

size = 100  # no of data samples
true_intercept = 0.5  # intercept of the regression line
true_slope = 2  # slop (coefficient) of x

x = np.linspace(0, 1, size)  # generate some random values for x in the given range (0,1)

# performs simple linear regression
# y = a + b*x
true_regression_line = true_intercept + true_slope * x

# we can't use true regression values for training the model
# therefore, add some random noise
y = true_regression_line + np.random.normal(scale=.3, size=size)

# split the data points into train and test split
x_train, x_test, y_train, y_test, true_y1, true_y2 = train_test_split(x, y, true_regression_line)

# we use a shared variable from theano to feed the x values into the model
# this is need for PPC
# when using the model for predictions we can set this shared variable to x_test
shared_x = shared(x_train)

# training the model
# model specifications in PyMC3 are wrapped in a with-statement
with pm.Model() as model:
    # Define priors
    x_coeff = pm.Normal('x', 0, sd=20)  # prior for coefficient of x
    intercept = pm.Normal('Intercept', 0, sd=20)  # prior for the intercept
    sigma = pm.HalfCauchy('sigma', beta=10)  # prior for the error term of due to the noise

    mu = intercept + tt.dot(shared_x, x_coeff)  # represent the linear regression relationship

    # Define likelihood
    likelihood = pm.Normal('y', mu=mu, sd=sigma, observed=y_train)

    # Inference!
    trace = pm.sample(1000, njobs=1)  # draw 3000 posterior samples using NUTS sampling

# predicting the unseen y values
# uses posterior predictive checks (PPC)
shared_x.set_value(x_test)  # let's set the shared x to the test dataset
ppc = pm.sample_ppc(trace, model=model, samples=1000)  # performs PPC
predictions = ppc['y'].mean(axis=0)  # compute the mean of the samples draws from each new y

# now you can measure the error
print("\n MSE of simple linear regression using bayesian : {0}\n".format(mean_squared_error(y_test, predictions)))