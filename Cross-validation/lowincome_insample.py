# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 10:39:27 2021

@author: Pranta
"""

import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
from matplotlib import figure
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared, Matern

print(__doc__)
df = pd.read_csv("LifeExpectancyData.csv", delimiter=None, header= 0)
countrylist=["Afghanistan", "Ethiopia", "Niger", "Sierra Leone", "Sudan", "Yemen, Rep."]
whichcount=1
def life_expectancy():
    y = df.Time.to_list()
    age = df.LifeExpectancy.to_list()
    country = df.Country.to_list()

    ys = np.asarray(y[0:9900]).reshape(-1, 1)
    ages = np.asarray(age[0:9900])
    countries = np.asarray(country[0:9900])
    return ys,ages, countries
Xr, yr, z  = life_expectancy()
plt.figure(figsize=(20, 8), dpi=300)
for name in countrylist:
    country = np.where(z == name)
    start = int(country[0][0])
    Xtemp = np.zeros(60)
    yTemp = np.zeros(60)
    print(country)
    n = 0
    for i in range(start, start + 60):
        Xtemp[n] = Xr[i]
        yTemp[n] = yr[i]
        n = n + 1
    XT = Xtemp[np.newaxis]
    X = XT.T
    y = yTemp

    X_train = X[:-20]
    X_test = X[-20:]
    print(X_train)

    # Split the targets into training/testing sets
    y_train = y[:-20]
    y_test = y[-20:]

    # Kernel with optimized parameters
    k1 = 50.0 ** 2 * RBF(length_scale=50)  # long term smooth rising trend
    k2 = 2.0 ** 2 * RBF(length_scale=0.1) \
         * ExpSineSquared(length_scale=1.0, periodicity=1.0,
                          periodicity_bounds="fixed")  # seasonal component
    # medium term irregularities
    k3 = 10 ** 2 * RationalQuadratic(length_scale=1, alpha=.50)
    k4 = 0.1 ** 2 * RBF(length_scale=0.3) * WhiteKernel(noise_level=0.05 ** 2,
                                                        noise_level_bounds=(1e-5, np.inf))  # noise terms
    k5 = 10 ** 2 * Matern(length_scale=.90, nu=1.5)
    k6 = 1 ** 2 * ExpSineSquared(length_scale=10.0, periodicity=5.0, periodicity_bounds="fixed")
    kernel = k1 + k3 + k4

    gp = GaussianProcessRegressor(kernel=kernel, alpha=0,
                                  normalize_y=True)
    print(X, y)
    gp.fit(X_train, y_train)

    print("\nLearned kernel: %s" % gp.kernel_)
    print("Log-marginal-likelihood: %.3f"
          % gp.log_marginal_likelihood(gp.kernel_.theta))

    X_ = np.linspace(X_train.min(), X_train.max() + 20, 1000)[:, np.newaxis]
    y_pred, y_std = gp.predict(X_, return_std=True)
    # Illustration
    plt.subplot(2,3,whichcount)
    plt.suptitle("(a) In-Sample Life Expectancy of Low-Income Countries",fontsize=28,fontweight='bold')
    r = random.random()
    b = random.random()
    g = random.random()
    col = (r, g, b)
    foo = ['o', 'v', '^', 's', 'p', '*']
    marker = random.choice(foo)
    funcyt = np.linspace(0, 100, 100)
    funct = np.linspace(2000, 2000, 100)
    plt.plot(funct, funcyt, alpha=0.5)
    plt.plot(X_, y_pred, '--', linewidth=4.0, label='Predicted Value')
    plt.plot(X, y, color='k', linewidth=4.0,
             label='Original Value'
             )
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    axes = plt.gca()
    axes.xaxis.label.set_size(18)
    axes.yaxis.label.set_size(18)
    # plt.fill_between(X_[:, 0], y_pred - y_std, y_pred + y_std,
    #                  alpha=0.6
    #                  # ,color=col
    #                  # ,label="95% Confidence Interval"
    #                  )
    plt.xlim(X_.min(), X_.max())
    plt.ylim(30, 90)
    plt.xlabel("Time", fontsize=20)
    plt.ylabel(r"LE (In Years)", fontsize=20)
    plt.legend(loc="upper left", prop={'size': 26})
    plt.title(r"%s" % name, fontsize=28)
    plt.tight_layout(pad=1.0)
    plt.grid()
    whichcount = whichcount + 1
plt.savefig('low.png')
plt.show()


