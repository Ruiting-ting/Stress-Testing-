import numpy as np
from scipy import stats
import statsmodels.api as sm
import pandas as pd
import numpy.linalg as npl
import matplotlib.pyplot as plt
import numpy.random as npr
import time
from tqdm import tqdm
from scipy.stats import norm
import scipy.optimize as opt


def fnDataImport(bDropNA=True):
    """Short summary.

    Parameters
    ----------
    bDropNA : type
        Description of parameter `bDropNA`.

    Returns
    -------
    type
        Description of returned object.

    """

    df = pd.read_excel('D:\QRM_Program\Research_Project\MarketData.xlsx', parse_dates=['Quarter'], index_col='Quarter')
    df.columns = ["GDP", "WTI", "HPI", "SMX", "ASCX"]

    if bDropNA:
        return df.dropna()
    else:
        return df

''' We use log returns to do the calibration, and prices for simulation. '''
def MJD_calibration(vTheta, Series, dt):

    # magical numbers: d is for the diffusion part, j is for the jumps
    mu_d = vTheta[0]
    sigma_d = np.exp(vTheta[1])
    mu_j = vTheta[2]
    sigma_j = vTheta[3]
    Lambda = vTheta[4]

    GDP = Series   # The series should be log returns

    for k in range(0, 20):
        mean = (mu_d - sigma_d**2 / 2 * dt) + (mu_j * k)
        std = (sigma_d ** 2 * dt + sigma_j**2 * k)
        xpdfs = norm.pdf(x=GDP, loc=mean, scale=np.sqrt(std))

        pk_denominator = (Lambda * dt) ** k / np.math.factorial(k)
        Pk = (pk_denominator * np.exp(-Lambda*dt))
        obj = - np.sum(np.log(xpdfs + Pk))
        # print(obj)

    return obj

''' We use log returns to do the calibration, and prices for simulation. '''

''' mReturn is prices data series; LogReturns is log returns data series'''

def jump_diffusion(mReturn, iSims, T, dt, log_corr, LogReturns):
    vTheta0 = [0.01, 2, 0.08, 0.02, 2]
    mParams = np.zeros((5, 5))
    #myfactr = 1e2
    for i in range(0, 5):
        res = opt.minimize( MJD_calibration, vTheta0, args=(
            LogReturns[:, i][~np.isnan(LogReturns[:, i])], dt), method='L-BFGS-B', options={'factr' : 10.0} )
        print(res.message)
        mParams[i, :] = res.x
    print(mParams)

    mu = mParams[:, 0]
    sigma = np.exp(mParams[:, 1])
    jumps_mu = mParams[:, 2]
    jumps_sigma = mParams[:, 3]
    Lambdas = mParams[:, 4]

    NAssets = mReturn.shape[1]
    # calculate amount of steps in simulations
    iSteps = int(T/dt)

    decomposition = np.linalg.cholesky(log_corr)
    simulated_paths = np.zeros([iSteps+1, iSims, NAssets])
    #simulated_paths[0, :, :] = df[-1]
    simulated_paths[0, :, :] = mReturn[-1]

    for sim in tqdm(range(iSims)):
        Z_1 = np.random.normal(0., 1., size=(iSteps + 1, NAssets))
        Z_2 = np.random.normal(0., 1., size=(iSteps + 1, NAssets))
        Poisson = np.random.poisson(lam=Lambdas*dt, size=(iSteps + 1, NAssets))

        Z_1_1 = Z_1 @ decomposition
        Z_2_1 = Z_2 @ decomposition

        for i in range(1, iSteps + 1):
            musigmaDelta = (mu - sigma**2/2) * dt
            sigmasqrtDelta = sigma * np.sqrt(dt)

            expPar1 = musigmaDelta + sigmasqrtDelta * Z_1_1[i, :]
            expPar2 = jumps_mu * Poisson[i, :] + jumps_sigma * np.sqrt(Poisson[i, :]) * Z_2_1[i, :]
            simulated_paths[i, sim, :] = simulated_paths[i-1, sim, :] * np.exp(expPar1 + expPar2)

    return simulated_paths


def Main():
    dfFull = fnDataImport(bDropNA=False)
    dfReturnFull = np.log(dfFull).diff()
    dfReturnFullClipped = dfReturnFull.copy()
    dfReturnFullClipped[['GDP', 'HPI']] = dfReturnFull[['GDP', 'HPI']].clip(
        lower=dfReturnFull['GDP'].quantile(0.01), upper=dfReturnFull['GDP'].quantile(0.99))

    mReturn = dfFull.values              # mReturn is prices data series; LogReturns is log returns data series
    LogReturns = dfReturnFull.values

    mCorr = dfReturnFull.corr().values
    # magic numbers
    T = 3  # simulate 3 years ahead
    dt = 1/4  # each step is one quarter
    iSims = 1000  # simulate iSim scenarios

    mS = jump_diffusion(mReturn, iSims, T, dt, mCorr, LogReturns)
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

    # we should compare prices changes with prices changes

    mHistorical = dfFull.pct_change(12).values   # Historical prices changes
    mSimulatedDistributions = np.zeros((iSims, mReturn.shape[1]))   # Simulated price changes
    for i in range(mReturn.shape[1]):
        mSimulatedDistributions[:, i] = ((mS[-1, :, i] - mS[0, :, i]) / mS[0, :, i])

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle('Simulating %i paths for %i assets' % (iSims, len(dfFull.columns)))
    columns = 2
    rows = 3
    list_assets = ["GDP", "WTI", "HPI", "SMX", "ASCX"]
    for i in range(1, 6):
        fig.add_subplot(rows, columns, i)
        plt.hist([mSimulatedDistributions[:, i-1], mHistorical[:, i-1]], color=['g', 'r'],
                 label=['Generated 3Y-change Asset '+list_assets[i-1],
                        'Historical 3Y-change Asset '+list_assets[i-1]], bins=40, density=True)
        plt.legend()
    plt.show()



if __name__ == '__main__':
    Main()
