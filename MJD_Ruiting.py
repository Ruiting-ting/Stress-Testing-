import numpy as np
from scipy import stats
import statsmodels.api as sm
import pandas as pd
import numpy.linalg as npl
import matplotlib.pyplot as plt
import numpy.random as npr
import time
import seaborn as sns
from tqdm import tqdm
from scipy.stats import norm
import scipy.optimize as opt
import math

def fnDataImport(bDropNA=True):

    df = pd.read_excel('D:\QRM_Program\Research_Project\MarketData.xlsx', parse_dates=['Quarter'], index_col='Quarter')
    df.columns = ["GDP", "WTI", "HPI", "SMX", "ASCX"]

    if bDropNA:
        return df.dropna()
    else:
        return df

###########################################################################
'''MJD_calibration'''
T = 12
Nsteps = 12
Delta_t =  T / Nsteps
vTheta = [0.22, 0.15, 0.08, 0.02, 20]

def MJD_calibration(vTheta, Series):
    mu_d = vTheta[0]
    sigma_d = np.exp(vTheta[1])
    mu_j = vTheta[2]
    sigma_j = vTheta[3]
    Lambda = vTheta[4]

    GDP = Series

    for k in range(0, 20):
        mean = (mu_d - sigma_d**2 / 2 * Delta_t) + (mu_j * k)
        std = (sigma_d ** 2 * Delta_t + sigma_j**2 * k)
        xpdfs = norm.pdf(x=GDP, loc=mean, scale=np.sqrt(std))

        pk_denominator = (Lambda * Delta_t) ** k / np.math.factorial(k)
        ln_Pk = (pk_denominator * np.exp(-Lambda*Delta_t))
        obj = - np.sum(np.log(xpdfs + ln_Pk))

    return obj

dfFull = fnDataImport(bDropNA=False)
dfReturnFull = np.log(dfFull).diff()
df_new = dfReturnFull.dropna()

Xs = np.zeros((5, 5))
for i in range(0,5):
    seriesI = df_new.iloc[:, i].values
    res= opt.minimize(MJD_calibration, vTheta, args=seriesI, method='Nelder-Mead')
    print(res.message)
    Xs[i,:] = res.x

###############################################################################################
'''MJD_simulation'''

def jump_diffusion (S, mu, sigma, Lambdas, Nsim, NAssets, T, Delta_t, Nsteps, log_corr, jumps_mu, jumps_sigma):

    decomposition = np.linalg.cholesky(log_corr)
    simulated_paths = np.zeros([Nsteps+1, Nsim, NAssets])
    simulated_paths[0, :,: ] = S

    for sim in tqdm(range(Nsim)):
        Z_1 = np.random.normal(0., 1., size=( Nsteps + 1, NAssets ))
        Z_2 = np.random.normal(0., 1., size=( Nsteps + 1, NAssets ))
        Poisson = np.random.poisson(lam=Lambdas*Delta_t, size=(Nsteps + 1, NAssets))

        Z_1_1 = Z_1 @ decomposition
        Z_2_1 = Z_2 @ decomposition


        for i in range(1, Nsteps + 1):
            musigmaDelta = (mu - sigma**2/2) * Delta_t
            sigmasqrtDelta = sigma * np.sqrt(Delta_t)

            expPar1 = musigmaDelta + sigmasqrtDelta * Z_1_1[i,:]
            expPar2 = jumps_mu * Poisson[i, :] + jumps_sigma * np.sqrt(Poisson[i, :]) * Z_2_1[i, :]
            simulated_paths[i, sim,: ] = simulated_paths[i-1, sim,: ] *  np.exp(expPar1 + expPar2)

    return simulated_paths

dfReturnFullClipped = df_new.copy()
dfReturnFullClipped[['GDP', 'HPI']] = dfReturnFull[['GDP', 'HPI']].clip(lower=dfReturnFull['GDP'].quantile(0.01), upper=dfReturnFull['GDP'].quantile(0.99))
#dfReturnFullClipped
mu = Xs[:,0]
sigma = np.exp(Xs[:,1])
sigma
jumps_mu = Xs[:,2]
jumps_mu
jumps_sigma = np.exp( Xs[:,3] )
jumps_sigma
Lambdas = Xs[:,4]
Lambdas

S = dfReturnFullClipped.iloc[-1].values
S
Nsim = 1000
NAssets = len(S)
T = 12
Nsteps = 12
Delta_t =  T / Nsteps
log_corr = df_new.corr()

mS = jump_diffusion (S, mu, sigma, Lambdas, Nsim, NAssets, T, Delta_t, Nsteps, log_corr, jumps_mu, jumps_sigma)

plt.figure(figsize=(12, 10))
plt.subplot(3, 2, 1)
plt.plot(mS[:, :, 0])

plt.subplot(3, 2, 2)
plt.plot(mS[:, :, 1])

plt.subplot(3, 2, 3)
plt.plot(mS[:, :, 2])

plt.subplot(3, 2, 4)
plt.plot(mS[:, :, 3])

plt.subplot(3, 2, 5)
plt.plot(mS[:, :, 4])
plt.show()

############################################################################
''' Comparison '''

mHistorical = dfFull.pct_change(12).values

fig = plt.figure(figsize=(8, 6))
fig.suptitle('Simulating %i paths for %i assets' % (Nsim, len(dfFull.columns)))
columns = 2
rows = 3
list_assets = ["GDP", "WTI", "HPI", "SMX", "ASCX"]
for i in range(1, 6):
    fig.add_subplot(rows, columns, i)
    plt.hist([np.sum(mS[-12:, :, i-1], axis=0), mHistorical[:, i-1]], color=['g', 'r'],
             label=['Generated 3Y-change Asset '+list_assets[i-1],
                    'Historical 3Y-change Asset '+list_assets[i-1]], bins=40, density=True)
    plt.legend()
plt.show()
