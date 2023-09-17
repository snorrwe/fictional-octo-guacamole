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
osl_estimate = regr.coef_[0]
print("OLS regression estimate", osl_estimate)
print("R2 score", r2_score(total_time, pred_total_time))
print("mean", mean)
print("stdev", np.std(measurements))

ax = measurements.plot.kde()
(ymin, ymax) = ax.get_ylim()
plt.vlines(osl_estimate, ymin, ymax, color="red")
plt.vlines(mean, ymin, ymax, color="blue")

plt.figure()

plt.scatter(X, measurements)

plt.figure()

plt.scatter(X, total_time, color="black")
plt.plot(X, pred_total_time, linewidth=3)
plt.show()
