# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 10:39:27 2021

@author: Pranta Biswas
"""

import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared, Matern

print(__doc__)
df = pd.read_csv("LifeExpectancyData.csv", delimiter=None, header= 0)

plt.figure(figsize=(20, 18), dpi=300)
def low_lifeexpectancy():
    countrylist=["Afghanistan", "Ethiopia", "Niger", "Sierra Leone", "Sudan", "Yemen, Rep."]
    def life_expectancy():
        y = df.Time.to_list()
        age = df.LifeExpectancy.to_list()
        country = df.Country.to_list()

        ys = np.asarray(y[0:10200]).reshape(-1, 1)
        ages = np.asarray(age[0:10200])
        countries = np.asarray(country[0:10200])
        return ys,ages, countries
    Xr, yr, z  = life_expectancy()
    plt.subplot(2, 1, 1)
    plt.grid()
    for name in countrylist:
        country= np.where(z == name)
        start=int(country[0][0])
        Xtemp=np.zeros(60)
        yTemp=np.zeros(60)
        print (name)
        n=0
        for i in range (start, start+60):
            Xtemp[n] = Xr[i]
            yTemp[n] = yr[i]
            n=n+1
        XT=Xtemp[np.newaxis]
        X=np.flip(XT.T)
        y=np.flip(yTemp)


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
        kernel = k1+k3+k4

        gp = GaussianProcessRegressor(kernel=kernel, alpha=0,
                                      normalize_y=True)
        gp.fit(X, y)

        print("\nLearned kernel: %s" % gp.kernel_)
        print("Log-marginal-likelihood: %.3f"
              % gp.log_marginal_likelihood(gp.kernel_.theta))
        lml=gp.log_marginal_likelihood(gp.kernel_.theta)
        labelname = name+" "+str(round(lml,3))
        X_ = np.linspace(X.min(), X.max() + 21, 1000)[:, np.newaxis]
        y_pred, y_std = gp.predict(X_, return_std=True)

        # Illustration
        # plt.figure(num=cnt)
        r = random.random()
        b = random.random()
        g = random.random()
        col = (r, g, b)
        foo = ['o', 'v', '^', 's','p','*']
        marker=random.choice(foo)
        plt.scatter(X, y,linewidth=3.0
                    ,marker=marker
                    ,label=name)

        plt.plot(X_, y_pred,color='k',linewidth=3.0)
        plt.fill_between(X_[:, 0], y_pred - y_std, y_pred + y_std,
                         alpha=0.6
                         # , color=col
                         )
        plt.tick_params(axis='x',labelsize=18)
        plt.tick_params(axis='y', labelsize=18)
        axes = plt.gca()
        axes.xaxis.label.set_size(18)
        axes.yaxis.label.set_size(18)
        plt.xlim(X_.min(), X_.max())
        plt.ylim(30,90)
        plt.xlabel("Time", fontsize= 20)
        plt.ylabel(r"Life Expectancy (In Years)", fontsize= 20)
        plt.legend(loc="lower right", prop={'size': 26})
        plt.title(r"(a) Predicted Life Expectancy of Low-Income Countries",fontweight='bold',fontsize=28)
        plt.tight_layout()
        plt.grid()

def mid_lifeexpectancy():
    countrylist=["Bangladesh", "India", "Philippines", "Sri Lanka", "Vietnam","Zimbabwe"]
    def life_expectancy():
        y = df.Time.to_list()
        age = df.LifeExpectancy.to_list()
        country = df.Country.to_list()

        ys = np.asarray(y[0:10200]).reshape(-1, 1)
        ages = np.asarray(age[0:10200])
        countries = np.asarray(country[0:10200])
        return ys,ages, countries
    Xr, yr, z  = life_expectancy()
    plt.subplot(2,1,2)
    plt.grid()
    for name in countrylist:
        country= np.where(z == name)
        start=int(country[0][0])
        Xtemp=np.zeros(60)
        yTemp=np.zeros(60)
        print (name)
        n=0
        for i in range (start, start+60):
            Xtemp[n] = Xr[i]
            yTemp[n] = yr[i]
            n=n+1
        XT=Xtemp[np.newaxis]
        X=np.flip(XT.T)
        y=np.flip(yTemp)


        k1 = 50.0 ** 2 * RBF(length_scale=50)  # long term smooth rising trend

        # medium term irregularities
        k3 = 10 ** 2 * RationalQuadratic(length_scale=1, alpha=.50)
        k4 = 0.1 ** 2 * RBF(length_scale=0.3) * WhiteKernel(noise_level=0.05 ** 2,
                           noise_level_bounds=(1e-5, np.inf))  # noise terms

        kernel = k1+k3+k4

        gp = GaussianProcessRegressor(kernel=kernel, alpha=0,
                                      normalize_y=True)
        gp.fit(X, y)

        print("\nLearned kernel: %s" % gp.kernel_)
        print("Log-marginal-likelihood: %.3f"
              % gp.log_marginal_likelihood(gp.kernel_.theta))
        lml=gp.log_marginal_likelihood(gp.kernel_.theta)
        labelname = name+" "+str(round(lml,3))
        X_ = np.linspace(X.min(), X.max() + 21, 1000)[:, np.newaxis]
        y_pred, y_std = gp.predict(X_, return_std=True)

        # Illustration
        # plt.figure(num=cnt)
        r = random.random()
        b = random.random()
        g = random.random()
        col = (r, g, b)
        foo = ['o', 'v', '^', 's','p','*']
        marker=random.choice(foo)
        plt.scatter(X, y,linewidth=3.0
                    ,marker=marker
                    ,label=name)
        plt.plot(X_, y_pred,color='k',linewidth=3.0)
        plt.fill_between(X_[:, 0], y_pred - y_std, y_pred + y_std,
                         alpha=0.6
                         # , color=col
                         )
        plt.tick_params(axis='x',labelsize=18)
        plt.tick_params(axis='y', labelsize=18)
        axes = plt.gca()
        axes.xaxis.label.set_size(18)
        axes.yaxis.label.set_size(18)
        plt.xlim(X_.min(), X_.max())
        plt.ylim(30,90)
        plt.xlabel("Time", fontsize= 20)
        plt.ylabel(r"Life Expectancy (In Years)", fontsize= 20)
        plt.legend(loc="lower right", prop={'size': 26})
        plt.title(r"(b) Predicted Life Expectancy of Lower-Middle-Income Countries",fontweight='bold',fontsize=28)
        plt.tight_layout()
        plt.grid()

low_lifeexpectancy()
mid_lifeexpectancy()
plt.savefig('low_lowmid.pdf')
plt.show()


