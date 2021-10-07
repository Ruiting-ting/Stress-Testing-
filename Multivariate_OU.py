import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from tqdm import tqdm
import pandas as pd


def fnDataImport(bDropNA=True):

    df = pd.read_excel("D:\QRM_Program\Research_Project\MarketData.xlsx", parse_dates=['Quarter'], index_col='Quarter')
    df.columns = ["GDP", "WTI", "HPI", "SMX", "ASCX"]

    if bDropNA:
        return df.dropna()
    else:
        return df


def fnOU_calibration(mReturn, dt):
    mParams = np.zeros(shape=(3, 5))
    for stock in range(mReturn.shape[1]):
        # fit ARIMA(1,0,0) model > AR(1)
        mod = sm.tsa.arima.ARIMA(mReturn[:, stock][~np.isnan(mReturn[:, stock])], order=(1, 0, 0))
        res = mod.fit()

        # return parameters
        alpha = res.params[1]
        beta = res.params[0]
        sd_epsilon = np.sqrt(res.params[2])

        # calculate parameters for OU model
        dLambda = -np.log(alpha) / dt
        dMu = beta/(1-alpha)
        dSigma = sd_epsilon * np.sqrt(-2*np.log(alpha) / dt*(1-alpha**2))

        print ("dLambda", dLambda, "dMu", dMu, "dSigma: ", dSigma)

        # put parameters in mParams array
        mParams[:, stock] = np.array([dLambda, dMu, dSigma])
    print(mParams)
    return mParams


def fnOU_simulation(T, dt, iSims, mReturn, mCorr):
    """Short summary.

    Parameters
    ----------
    T : type
        Description of parameter `T`.
    dt : type
        Description of parameter `dt`.
    iSims : type
        Description of parameter `iSims`.
    mReturn : type
        Description of parameter `mReturn`.
    mCorr : type
        Description of parameter `mCorr`.

    Returns
    -------
    Matrix
        S - a (steps+1)-by-nsims-by-nassets 3-dimensional matrix where
        each row represents a time step, each column represents a
        seperate simulation run and each 3rd dimension represents a
        different asset.
    """

    # return calibrated parameters from function based on data given
    mParams = fnOU_calibration(mReturn, dt)  # [dLambda, dMu, dSigma]
    vLambda = mParams[0, :]
    vMu = mParams[1, :]
    vSigma = mParams[2, :]

    # S_0 should be equal to the last observed variable
    S_0 = mReturn[-1]

    # calculate amount of steps in simulations
    iSteps = int(T/dt)

    mS = np.zeros([iSteps + 1, iSims, len(S_0)])
    mS[0, :, :] = S_0

    for sim in tqdm(range(iSims)):
        mDW = np.random.multivariate_normal(np.zeros(len(S_0)), mCorr, iSteps + 1)

        for i in range(1, iSteps + 1):
            mS[i, sim, :] = mS[i-1, sim, :] * np.exp(-vLambda*dt) + vMu*(1-np.exp(-vLambda*dt)) + \
                vSigma*(np.sqrt((1-np.exp(-2*vLambda*dt))/(2*vLambda)))*mDW[i]

    return mS


def Main():
    # load all data and also changes
    dfFull = fnDataImport(bDropNA=False)

    # calculate log differences for additivity later on
    dfReturnFull = np.log(dfFull).diff()

    # clip extreme 1% and 99% values to better approximate "real-world"
    dfReturnFullClipped = dfReturnFull.copy()
    dfReturnFullClipped[['GDP', 'HPI']] = dfReturnFull[['GDP', 'HPI']].clip(lower=dfReturnFull['GDP'].quantile(
        0.01), upper=dfReturnFull['GDP'].quantile(0.99))

    # get historical 3 year changes/ 12 quarters
    mHistorical = dfFull.pct_change(12).values

    # calculate correlation matrix and changes matrix
    mCorr = dfReturnFull.corr().values
    mReturn = dfReturnFullClipped.values

    # magic numbers
    T = 12  # simulate 12 quarters ahead
    dt = 1  # each step is one quarter (20 working days)
    iSims = 5  # simulate iSim scenarios
    mS = fnOU_simulation(T, dt, iSims, mReturn, mCorr)

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Simulating %i paths for %i assets' % (iSims, len(dfFull.columns)))
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


if __name__ == '__main__':
    Main()
