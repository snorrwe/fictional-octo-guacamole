import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score

# create random test data, these should be output by the benchmarks
N = 100
measurements = np.random.normal(37, 10, N)
measurements = pd.Series(measurements)


# for each index (i) total_time[i] = total time spent so far (sum(measurements[0..i]))
total_time = np.cumsum(measurements)
X = np.arange(N, dtype=float).reshape(-1, 1)

regr = linear_model.LinearRegression()

regr.fit(X, total_time)

pred_total_time = regr.predict(X)

mean = np.average(measurements)
print("OLS regression estimate", regr.coef_[0])
print("R2 score", r2_score(total_time, pred_total_time))
print("mean", mean)
print("stdev", np.std(measurements))

measurements.plot.kde()

plt.figure()

plt.scatter(X, measurements)

plt.figure()

plt.scatter(X, total_time, color="black")
plt.plot(X, pred_total_time, linewidth=3)
plt.show()
