import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd


def fnDataImport(bDropNA=True):

    df = pd.read_excel("data/MarketData.xlsx", parse_dates=['Quarter'], index_col='Quarter')
    df.columns = ["GDP", "WTI", "HPI", "SMX", "ASCX"]

    if bDropNA:
        return df.dropna()
    else:
        return df


# def fnABM_Calibriation(mReturn):
#     mParams = np.zeros(shape=(2, 5))
#     for stock in range(mReturn.shape[1]):
#         endog = mReturn[:, stock][~np.isnan(mReturn[:, stock])]
#         exog = np.ones(len(endog))
#         model = sm.OLS(endog, exog).fit()
#         dMean = model.params[0]
#         dSigma = np.var(model.resid)
#         mParams[:, stock] = np.array([dMean, dSigma])
#
#     return mParams

def fnABM_calibration(mReturn):
    """Short summary.

    Parameters
    ----------
    mReturn : type
        Description of parameter `mReturn`.
    dt : type
        Description of parameter `dt`.

    Returns
    -------
    type
        Description of returned object.

    """
    mParams = np.zeros(shape=(2, 5))
    for stock in range(mReturn.shape[1]):
        # calculate mean
        dMean = np.mean(mReturn[:, stock][~np.isnan(mReturn[:, stock])])
        dVar = np.var(mReturn[:, stock][~np.isnan(mReturn[:, stock])])
        # put parameters in mParams array
        mParams[:, stock] = np.array([dMean, dVar])
    return mParams


def ABM_simulation(mReturn, mPrices, mCorr, dt, T, iSims):
    # return calibrated parameters from function based on data given
    mParams = fnABM_calibration(mReturn)  # [dLambda, dMu, dSigma]
    vMu = mParams[0, :]
    vSigma = np.sqrt(mParams[1, :])

    # get the number of assets
    iAssets = mReturn.shape[1]

    # calculate amount of steps in simulations
    iSteps = int(T/dt)

    # pre-allocate the output
    mS = np.zeros([iSteps + 1, iSims, iAssets])
    mS[0, :, :] = mPrices[-1, :]
    # generate correlated random sequences and paths
    for sim in tqdm(range(iSims)):
        # generate correlated random sequence
        mDW = np.random.multivariate_normal(np.zeros(iAssets), mCorr, iSteps + 1)

        for i in range(1, iSteps+1):
            mS[i, sim, :] = mS[i-1, sim, :] + vMu*dt + vSigma * np.sqrt(dt) * mDW[i]
    return mS


def Main():
    # load all data and also changes
    dfFull = fnDataImport(bDropNA=False)

    # calculate log differences for additivity later on
    dfReturnFull = (dfFull).diff()

    # calculate correlation matrix and changes matrix
    mCorr = dfReturnFull.corr().values
    mReturn = dfReturnFull.values
    mPrices = dfFull.values

    # magic numbers
    T = 12  # simulate 12 quarters ahead
    dt = 1  # each step is one quarter (20 working days)
    iSims = 10000  # simulate iSim scenarios
    mS = ABM_simulation(mReturn, mPrices, mCorr, dt, T, iSims)

    # get historical 3 year changes/ 12 quarters
    mHistorical = dfFull.pct_change(12).values
    mSimulatedDistributions = np.zeros((iSims, mPrices.shape[1]))
    for i in range(mPrices.shape[1]):
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
